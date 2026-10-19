"""Exercise normal authenticated UI controls, without reading or exporting credentials."""
import json,time
from pathlib import Path
from playwright.sync_api import sync_playwright
out=Path('/opt/agillm43_netserve/bintf_telemetry/browser-generation-acceptance.json')
report={'started_at':time.time(),'checks':[],'page_errors':[]}
with sync_playwright() as p:
    browser=p.chromium.launch()
    page=browser.new_page(ignore_https_errors=True,viewport={'width':1440,'height':1000})
    page.set_default_timeout(90000)
    page.on('pageerror',lambda e:report['page_errors'].append(str(e)))
    try:
        page.goto('https://codex.opentransformers.online/agillm-ui',wait_until='domcontentloaded',timeout=20000)
        page.wait_for_timeout(6000)
        page.locator('#modelseg button[data-model="standard"]').click()
        for mode in ['ar','sat_fixed','sat_variable','nat']:
            page.locator('button[data-mode-key="'+mode+'"]').click()
            page.locator('#len').evaluate("e=>{e.value='8';e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}));}")
            page.locator('#doc').fill('The capital of France is')
            started=time.perf_counter()
            with page.expect_response(lambda r:r.request.method=='POST',timeout=30000) as pending:
                page.locator('#go').click()
            response=pending.value
            response.finished()
            data=response.text()
            result={'mode':mode,'route':response.url.split('?')[0],'status':response.status,'wall_s':round(time.perf_counter()-started,3),'response_bytes':len(data.encode()),'editor_text':page.locator('#doc').inner_text(),'response_tail':data[-1800:]}
            report['checks'].append(result)
            out.write_text(json.dumps(report,indent=2))
            print(json.dumps(result),flush=True)
            assert response.ok and '"ok": false' not in data, 'Generation returned an error'
            page.wait_for_timeout(300)
        report['ok']=True
    except Exception as exc:
        report['ok']=False;report['error']=type(exc).__name__+': '+str(exc)
    finally:
        report['finished_at']=time.time();out.write_text(json.dumps(report,indent=2));browser.close()
print(json.dumps({'ok':report['ok'],'checks':len(report['checks']),'error':report.get('error'),'path':str(out)}))
