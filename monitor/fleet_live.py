"""Fresh per-model fleet evidence, separate from preserved historical curves and serving."""
from __future__ import annotations
import copy, json, math, re, time
from pathlib import Path
PATH=Path("/opt/agillm43_netserve/fleet_training_status.json")
FRESH_PROBE_S=100
FRESH_TRAIN_S=120
def number(v):
    return v if isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) else None
def age(ts,now):
    n=number(ts)
    return max(0,now-n) if n is not None else None
def model_view(raw,now):
    d=copy.deepcopy(raw) if isinstance(raw,dict) else {}
    observed=number(d.get("observed_at"))
    source=number(d.get("source_observed_at"))
    d["age_s"]=age(observed,now);d["source_age_s"]=age(source,now)
    transport=bool(d.get("ok") and observed and -10<=now-observed<=FRESH_PROBE_S)
    completed=d.get("status")=="completed"
    d["fresh"]=bool(transport and d.get("fresh") and (completed or (source and -10<=now-source<=FRESH_TRAIN_S)))
    d["ok"]=transport
    if not transport:
        d["last_known_status"]=d.get("status")
        d["status"]="stale" if observed else "unavailable"
    elif d.get("running") and not d["fresh"]:
        d["status"]="stale"
    if not d["fresh"] or completed:
        for key in ("raw_tok_s","median_tok_s","wall_tok_s","median_step_s"): d[key]=None
    d["log_age_s"]=age(d.get("log_mtime"),now)
    d["freshness_basis"]="completed state verified at observed_at" if completed and transport else "source event age and verified trainer process"
    return d
def snapshot(now=None):
    now=time.time() if now is None else now
    try: raw=json.loads(PATH.read_text())
    except (OSError,ValueError): raw={}
    models=raw.get("models") or {}
    out={m:model_view(models.get(m),now) for m in ("standard","rpv16")}
    stamps=[d["observed_at"] for d in out.values() if number(d.get("observed_at")) is not None]
    return {"schema":"agillm.fleet.telemetry.v1","ok":any(d.get("ok") for d in out.values()),
            "models":out,"generated_at":max(stamps) if stamps else None,"interval_s":30,
            "checked_at":min(stamps) if stamps else None,"serving":serving_views(raw.get("serving") or {},now)}
def serving_views(raw,now):
    views={}
    for model in ("standard","rpv16"):
        h=copy.deepcopy(raw.get(model) or {})
        stamp=number(h.get("observed_at"))
        h["age_s"]=age(stamp,now)
        h["fresh"]=bool(stamp and -10<=now-stamp<=FRESH_PROBE_S)
        if not h["fresh"]: h.update(ok=False,ready=False)
        ckpt=h.get("ckpt") or h.get("checkpoint") or (h.get("model") or {}).get("checkpoint")
        model_meta=h.get("model") if isinstance(h.get("model"),dict) else {}
        step=h.get("step") or h.get("serving_step") or model_meta.get("step")
        match=re.search(r"(?:step[_-]?)(\d+)",str(ckpt),re.I)
        h["serving_step"]=step if step is not None else int(match.group(1)) if match else None
        h["serving_checkpoint"]=ckpt
        views[model]=h
    return views
def area(d):
    return {"state":d.get("status","unavailable"),"fresh":d.get("fresh",False),"instance_id":d.get("instance_id"),
            "checked_at":d.get("observed_at"),"source_observed_at":d.get("source_observed_at"),
            "age_s":d.get("age_s"),"step":d.get("step"),"running":d.get("running"),
            "summary":(str(round((number(d.get("target_tokens")) or 0)/1e9,3))+"B target complete; trainer stopped" if d.get("status")=="completed" else
                       "Training on "+str(d.get("gpu_name")) if d.get("status")=="training" else
                       "Current trainer evidence is "+str(d.get("status","unavailable")))}
def health():
    fleet=snapshot();s,r=fleet["models"]["standard"],fleet["models"]["rpv16"]
    return {"ok":fleet["ok"],"overall":"ok" if s.get("fresh") and r.get("fresh") else "warn",
            "checked_at":fleet["checked_at"],"generated_at":fleet["generated_at"],"polled_at":fleet["generated_at"],
            "areas":{"trainer":area(s),"standard":area(s),"rpv16":area(r)},
            "notes":["Standard and RPV16 are separate model lineages. Training telemetry does not prove serving promotion."],
            "latest":{"standard":s.get("step"),"rpv16":r.get("step")},"serving":fleet["serving"]}
def compact(d):
    return {k:v for k,v in d.items() if k not in ("recent_history","heldout_recent_history","tail","val","objective_history","anchor_history","val_history","log_tail")}
def _objective_rows(v):
    # AGI44_OBJECTIVE_TELEMETRY_20261002: heartbeat rows (mixed local loss, objective "dblock") plus per-objective DBlock rows
    # (objective ar/sat/nat; global_anchor=True = full-stack AR anchor CE) parsed read-only from the trainer log.
    rows=list(v.get("recent_history") or [])+[dict(r) for r in (v.get("objective_history") or []) if isinstance(r,dict)]
    rows=[r for r in rows if isinstance(r,dict) and number(r.get("step")) is not None]
    rows.sort(key=lambda r:r["step"])
    return rows
def training(base):
    d=copy.deepcopy(base)
    f=snapshot();s,r=f["models"]["standard"],f["models"]["rpv16"]
    if "source" in d: d["historical_curve_source"]=d.get("source")
    d.update(ok=f["ok"],source="current-vast-fleet-with-preserved-historical-curves",
             generated_at=f["generated_at"],polled_at=f["generated_at"],health=health(),
             standard_live=compact(s),rpv16_live=compact(r),gb10_live=compact(r),live=live(f),
             live_history={m:{"lineage":v.get("lineage"),"instance_id":v.get("instance_id"),"observed_at":v.get("observed_at"),
                              "source_observed_at":v.get("source_observed_at"),"rows":_objective_rows(v),"anchors":v.get("anchor_history") or [],"val":v.get("val_history") or [],"accum":v.get("objective_history_accum"),
                              "fresh":v.get("fresh"),"status":v.get("status")} for m,v in f["models"].items()},
             historical_curves_note="Existing history, standard_history, probes and curves are preserved lineage history; use live_history for current-host recent rows.")
    # Never let an old GB10/FLOP cache masquerade as current host telemetry.
    d.pop("compute_status",None)
    d["latest"]={k:r.get(k) for k in ("step","seen_tokens","loss","objective","lr")}
    d["latest"]["ts"]=r.get("source_observed_at")
    d["latest"]["tok_s"]=r.get("median_tok_s")
    d["latest"]["stale"]=not r.get("fresh")
    d["reason"]=None if f["ok"] else "Current fleet telemetry unavailable; preserved history is not live."
    # 2026-10-09: "degraded" was inherited verbatim from the retired legacy snapshot, so a fully
    # fresh fleet still advertised itself as degraded. Now the flag describes the live feed, the
    # preserved GB10 curves are labelled as the archive they are, and their age is explicit.
    live_rows=sum(len(v.get("rows") or []) for v in (d.get("live_history") or {}).values())
    d["degraded"]=not f["ok"]
    d["curves_available"]=live_rows>0
    d["live_curve_rows"]=live_rows
    hs=d.get("history_state") or {}
    ref=hs.get("refreshed_at")
    if isinstance(ref,(int,float)) and ref>0:
        d["preserved_history_age_days"]=round((time.time()-float(ref))/86400.0,2)
        d["preserved_history_note"]=("Preserved GB10 lineage curves, archived %.1f days ago; "
                                     "current training is in live_history." % d["preserved_history_age_days"])
    else:
        d["preserved_history_note"]="No preserved lineage curves are present."
    return d
def live(f=None):
    f=f or snapshot();s,r=copy.deepcopy(f["models"]["standard"]),copy.deepcopy(f["models"]["rpv16"])
    result={"schema":"agillm.fleet.live-telemetry.v2","ok":f["ok"],"probe_ts":f["generated_at"],"interval_s":30,
            "age_s":age(f["generated_at"],time.time()),"standard":s,"rpv16":r,"hosts":{},
            "gpu":r.get("gpu"),"host":r.get("host"),"lr_override":None,"heldout_live":{"history":[]}}
    for m,v in f["models"].items(): result["hosts"][m]={"instance_id":v.get("instance_id"),"gpu":v.get("gpu"),"host":v.get("host"),"observed_at":v.get("observed_at"),"fresh":v.get("fresh")}
    s["seen_tok"]=s.get("seen_tokens");s["loss_last"]=s.get("loss")
    s["tail"]={"cols":["step","seen_tok","loss","lr","raw_tok_s"],"rows":[[x.get("step"),x.get("seen_tokens"),x.get("loss"),x.get("lr"),x.get("tok_s")] for x in s.get("recent_history") or []]}
    s["val"]={"cols":["step","ce"],"rows":[[x.get("step"),x.get("ce")] for x in s.get("heldout_recent_history") or []]}
    r["last_checkpoint"]=r.get("last_checkpoint_event")
    for model in (s,r):
        model.pop("recent_history",None);model.pop("heldout_recent_history",None)
        for _k in ("objective_history","anchor_history","val_history","log_tail"): model.pop(_k,None)
    return result
def checkpoint(base):
    d=copy.deepcopy(base); f=snapshot()
    d["latest_trained"]={m:{**(v.get("last_checkpoint_event") or {}),"model":v.get("model"),"lineage":v.get("lineage"),
                             "instance_id":v.get("instance_id"),"observed_at":v.get("observed_at"),
                             "fresh":v.get("fresh"),"status":v.get("status")} for m,v in f["models"].items()}
    # Top-level checkpoint means latest Standard training artifact. Serving remains separate.
    standard=f["models"]["standard"]
    latest=standard.get("last_checkpoint_event") or {}
    d.update(ok=bool(standard.get("ok")),model_id="standard",model="AGILLM Standard 1.1B",branch=standard.get("lineage"),
             target_tokens=standard.get("target_tokens"),trainer_live_step=standard.get("step") if standard.get("running") and standard.get("fresh") else None,
             source=standard.get("source"),step=latest.get("step"),seen_tok=latest.get("seen_tokens"),
             checkpoint_name=Path(str(latest.get("path") or "")).name,checkpoint_path=latest.get("path"),
             checkpoint_sha256=latest.get("sha256"),saved_at_unix=latest.get("ts"),
             polled_at=standard.get("observed_at"),status=standard.get("status"),fresh=standard.get("fresh"),
             health=f["serving"]["standard"],serving=f["serving"])
    d["standard_live"]=compact(f["models"]["standard"])
    d["rpv16_live"]=compact(f["models"]["rpv16"])
    d["training_observed_at"]=f["generated_at"]
    d["training_note"]="Latest trained checkpoints are independent of the actual serving health and checkpoint."
    return d


_PNG_CACHE={"at":0,"bytes":b""}
def png():
    """Current-host recent curves; archived full histories stay in the JSON/UI."""
    now=time.time()
    if _PNG_CACHE["bytes"] and now-_PNG_CACHE["at"]<25: return _PNG_CACHE["bytes"]
    from PIL import Image, ImageDraw, ImageFont
    from io import BytesIO
    from datetime import datetime,timezone
    f=snapshot(); im=Image.new("RGB",(1440,880),(12,17,24));draw=ImageDraw.Draw(im)
    try:
        font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",20)
        title=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",28)
    except OSError: font=title=ImageFont.load_default()
    draw.text((30,20),"AGILLM - current-host training evidence",font=title,fill=(235,241,248))
    draw.text((30,60),"Separate model lineages; raw training loss for one objective per chart. This is not held-out evaluation.",font=font,fill=(159,176,196))
    for i,model in enumerate(("standard","rpv16")):
        d=f["models"][model]; y=108+i*372
        allrows=d.get("recent_history") or []
        obj=(allrows[-1].get("objective") if allrows else None)
        rows=[r for r in allrows if r.get("objective")==obj and number(r.get("step")) is not None and number(r.get("loss")) is not None]
        draw.text((30,y),str(d.get("model",model))+" | "+str(d.get("gpu_name","unknown"))+" | "+str(d.get("status","unavailable")),font=title,fill=(235,241,248))
        stamp=d.get("source_observed_at")
        stamptext=datetime.fromtimestamp(stamp,timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC") if number(stamp) else "unavailable"
        draw.text((30,y+39),"Objective "+str(obj)+" | source "+stamptext+" | observed age "+str(round(d["age_s"],1) if d.get("age_s") is not None else "unknown")+" s",font=font,fill=(159,176,196))
        x0,x1,y0,y1=90,1398,y+93,y+301
        draw.rectangle((x0,y0,x1,y1),outline=(49,65,82))
        if len(rows)<2:
            draw.text((x0+20,y0+50),"Insufficient measured rows for a curve",font=font,fill=(220,180,100));continue
        xs=[r["step"] for r in rows];ys=[r["loss"] for r in rows]
        low,high=min(ys),max(ys);span=max(high-low,0.02);lo,hi=low-span*.08,high+span*.08
        def X(v):return x0+(v-min(xs))/max(1,max(xs)-min(xs))*(x1-x0)
        def Y(v):return y1-(v-lo)/(hi-lo)*(y1-y0)
        for n in range(5):
            value=lo+(hi-lo)*n/4; yy=Y(value)
            draw.line((x0,yy,x1,yy),fill=(28,40,55));draw.text((8,yy-10),("%.2f"%value),font=font,fill=(159,176,196))
        draw.line([(X(r["step"]),Y(r["loss"])) for r in rows],fill=(70,183,215) if model=="standard" else (151,132,245),width=2)
        draw.text((x0,y1+8),f"step {min(xs):,}",font=font,fill=(159,176,196))
        draw.text((x1-270,y1+8),f"step {max(xs):,} | {len(rows)} rows",font=font,fill=(159,176,196))
    out=BytesIO();im.save(out,format="PNG");b=out.getvalue()
    _PNG_CACHE.update(at=now,bytes=b);return b
