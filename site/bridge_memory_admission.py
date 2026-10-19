"""Bound admission of inference model loads, without cancelling requests or changing cgroup limits."""
import os
import time
from pathlib import Path

GIB=1 << 30
VERSION='bridge-memory-admission-v2-reclaimable-cache-20261010'

def decide(snapshot, peak_bytes=6*GIB, host_reserve_bytes=2*GIB, group_reserve_bytes=384*(1<<20)):
    """Read-only conservative admission. Existing workers/requests remain untouched."""
    result={'version':VERSION,'allowed':False,'reason':'memory_observation_unavailable',
            'new_child_peak_bytes':peak_bytes,'host_reserve_bytes':host_reserve_bytes,
            'group_reserve_bytes':group_reserve_bytes,**snapshot}
    if not snapshot.get('observed'):return result
    if snapshot['host_available_bytes'] < peak_bytes+host_reserve_bytes:
        result['reason']='host_reserve';return result
    if snapshot['psi_full_avg10']>=5.0 or snapshot['psi_full_avg60']>=5.0:
        result['reason']='host_memory_pressure';return result
    ceiling=snapshot.get('group_max_bytes')
    if ceiling is not None and ceiling-snapshot['group_current_bytes']<peak_bytes+group_reserve_bytes:
        result['reason']='inference_group_headroom';return result
    result.update(allowed=True,reason='capacity_available');return result

def _reclaimable_file_bytes(node):
    """Clean, unmapped page cache charged to a cgroup (checkpoint files hashed at startup, raw weight
    cache left behind by a retired worker). The kernel reclaims it under memory.max, so it must not
    count as consumed memory. Mapped weights, dirty/writeback pages and shmem stay counted."""
    stat={}
    for line in (node/'memory.stat').read_text().splitlines():
        key,_,value=line.partition(' ')
        if key in ('file','file_mapped','file_dirty','file_writeback','shmem'):stat[key]=int(value)
    return max(0,stat.get('file',0)-stat.get('file_mapped',0)-stat.get('file_dirty',0)-stat.get('file_writeback',0)-stat.get('shmem',0))

def _charged_bytes(node):
    return max(0,int((node/'memory.current').read_text())-_reclaimable_file_bytes(node))

def snapshot():
    result={'observed':False,'at_unix':time.time()}
    try:
        memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','MemTotal:'))}
        pressure=next(line for line in Path('/proc/pressure/memory').read_text().splitlines() if line.startswith('full '))
        psi=dict(item.split('=') for item in pressure.split()[1:])
        rel=next(line.split('::',1)[1] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
        root=Path('/sys/fs/cgroup').resolve();group=(root/rel.lstrip('/')).resolve()
        if not group.is_relative_to(root):raise ValueError('Invalid cgroup path')
        ceilings=[];available=[]
        raw_current=int((group/'memory.current').read_text())
        reclaimable=_reclaimable_file_bytes(group)
        current=max(0,raw_current-reclaimable)
        # Include any parent slice limit, not just this individual service.
        node=group
        while node!=root:
            path=node/'memory.max'
            if path.exists():
                raw=path.read_text().strip()
                if raw!='max':
                    cap=int(raw);used=_charged_bytes(node);available.append(max(0,cap-used))
            node=node.parent
        effective_max=current+min(available) if available else None
        result.update(observed=True,host_available_bytes=memory['MemAvailable'],host_total_bytes=memory['MemTotal'],
                      psi_full_avg10=float(psi['avg10']),psi_full_avg60=float(psi['avg60']),
                      group_current_bytes=current,group_raw_current_bytes=raw_current,group_reclaimable_file_bytes=reclaimable,
                      group_max_bytes=effective_max,
                      inherited_headroom_bytes=min(available) if available else None)
    except (OSError,ValueError,KeyError,StopIteration) as exc:
        result['observation_error']=type(exc).__name__
    return result

def status():
    return decide(snapshot())

_last_warning=0.0

def require_capacity():
    global _last_warning
    import json
    decision=status()
    if not decision['allowed']:
        now=time.monotonic()
        if now-_last_warning>=60:
            print('[bridge] model-load admission '+json.dumps(decision,sort_keys=True),flush=True)
            _last_warning=now
        raise RuntimeError('model_load_deferred:'+decision['reason'])
    return decision

if __name__=='__main__':
    import json
    base={'observed':True,'host_available_bytes':16*GIB,'psi_full_avg10':0.,'psi_full_avg60':0.,'group_current_bytes':64*(1<<20),'group_max_bytes':8*GIB}
    checks=[]
    def check(name,condition):
        checks.append({'name':name,'passed':bool(condition)})
    check('first_child_fits',decide(base)['allowed'])
    check('second_model_blocked_not_requests',decide({**base,'group_current_bytes':4*GIB})['reason']=='inference_group_headroom')
    check('ample_capacity_preserves_two_workers',decide({**base,'group_current_bytes':4*GIB,'group_max_bytes':12*GIB})['allowed'])
    check('host_reserve',decide({**base,'host_available_bytes':7*GIB})['reason']=='host_reserve')
    check('psi_short',decide({**base,'psi_full_avg10':5.1})['reason']=='host_memory_pressure')
    check('psi_sustained',decide({**base,'psi_full_avg60':5.1})['reason']=='host_memory_pressure')
    check('missing_observation_defers_load',not decide({'observed':False})['allowed'])
    check('unlimited_group_still_checks_host',decide({**base,'group_max_bytes':None})['allowed'])
    check('cgroup_reserve',not decide({**base,'group_max_bytes':6*GIB})['allowed'])
    check('input_not_mutated',base['group_current_bytes']==64*(1<<20))
    live=status()
    check('live_reclaimable_not_counted',(not live.get('observed')) or live['group_current_bytes']==max(0,live['group_raw_current_bytes']-live['group_reclaimable_file_bytes']))
    print(json.dumps({'ok':all(x['passed'] for x in checks),'tests':checks,'actual_memory':live},indent=2))
    if not all(x['passed'] for x in checks):raise SystemExit(1)
