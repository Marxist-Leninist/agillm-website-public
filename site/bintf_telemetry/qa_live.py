import json
from pathlib import Path
from playwright.sync_api import sync_playwright
out=Path('/opt/agillm43_netserve/bintf_telemetry/qa')
r={'errors':[], 'http_errors':[], 'failed_requests':[]}
with sync_playwright() as p:
    b=p.chromium.launch()
    q=b.new_page(ignore_https_errors=True,viewport={'width':1440,'height':1000})
    q.set_default_timeout(5000)
    q.on('pageerror',lambda e:r['errors'].append(str(e)))
    q.on('requestfailed',lambda x:r['failed_requests'].append({'url':x.url,'error':x.failure}))
    q.on('response',lambda x:r['http_errors'].append({'url':x.url,'status':x.status}) if x.status>=400 else None)
    try:
        q.goto('https://codex.opentransformers.online/agillm-ui',wait_until='domcontentloaded',timeout=20000)
        q.wait_for_timeout(10000)
        r['monitor']=q.locator('#agillmFleetMonitor').inner_text()
        r['desktop_editor']=q.locator('#doc').bounding_box()
        r['cards']=q.locator('.af-card').count()
        q.screenshot(path=str(out/'desktop-final.png'))
        q.locator('#modelseg button[data-model="bintf4b"]').click()
        q.wait_for_timeout(800)
        q.set_viewport_size({'width':390,'height':844})
        q.wait_for_timeout(800)
        r['mobile_monitor']=q.locator('#agillmFleetMonitor').bounding_box()
        r['mobile_binary']=q.locator('.af-card[data-model="bintf4b"]').bounding_box()
        r['mobile_editor']=q.locator('#doc').bounding_box()
        r['binary_selected']=q.locator('.af-card[data-model="bintf4b"]').get_attribute('data-selected')
        q.screenshot(path=str(out/'mobile-final.png'))
        q.locator('.training-heading .telemetry-hide-btn').click()
        q.wait_for_timeout(300)
        r['collapsed']=q.locator('#agillmTelemetry').get_attribute('class')
        r['editor_after_hide']=q.locator('#doc').bounding_box()
        q.screenshot(path=str(out/'mobile-focused-final.png'))
    except Exception as e:
        r['test_error']=str(e)
    finally:
        b.close()
        (out/'report-final.json').write_text(json.dumps(r,indent=2))
print(json.dumps(r))
