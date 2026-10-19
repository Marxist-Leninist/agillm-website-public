"""Browser smoke test of the deployed AGILLM UI, without changing saved user data."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

out=Path('/opt/agillm43_netserve/bintf_telemetry/qa')
out.mkdir(exist_ok=True)
report={'errors':[], 'http_errors':[], 'layouts':{}}
with sync_playwright() as p:
    browser=p.chromium.launch()
    page=browser.new_page(ignore_https_errors=True,viewport={'width':1440,'height':1000})
    page.on('pageerror',lambda error:report['errors'].append(str(error)))
    page.on('response',lambda response:report['http_errors'].append({'status':response.status,'url':response.url}) if response.status>=400 else None)
    page.goto('https://codex.opentransformers.online/agillm-ui',wait_until='domcontentloaded',timeout=20000)
    page.wait_for_timeout(4500)
    report['title']=page.title()
    report['monitor']=page.locator('#agillmFleetMonitor').inner_text()
    report['card_count']=page.locator('.af-card').count()
    report['layouts']['desktop']={}
    for selector in ['#workspace','#agillmTelemetry','#docwrap','#doc','#modelseg']:
        element=page.locator(selector)
        report['layouts']['desktop'][selector]=element.bounding_box() if element.count() else None
    page.screenshot(path=str(out/'desktop.png'))
    page.locator('#modelseg button[data-model="bintf4b"]').click()
    page.wait_for_timeout(1500)
    report['binary_selected']=page.locator('.af-card[data-model="bintf4b"]').get_attribute('data-selected')
    page.locator('.af-card[data-model="bintf4b"] summary').click()
    report['binary_details']=page.locator('.af-card[data-model="bintf4b"]').inner_text()
    page.screenshot(path=str(out/'binary.png'))
    page.set_viewport_size({'width':390,'height':844})
    page.wait_for_timeout(1200)
    report['layouts']['mobile']={}
    for selector in ['#workspace','#agillmTelemetry','#docwrap','#doc','#modelseg']:
        element=page.locator(selector)
        report['layouts']['mobile'][selector]=element.bounding_box() if element.count() else None
    page.screenshot(path=str(out/'mobile.png'))
    page.locator('#hdrTelemetryHide').click()
    page.wait_for_timeout(300)
    report['telemetry_hide_works']=not page.locator('#agillmTelemetry').is_visible()
    report['editor_visible_with_hidden_telemetry']=page.locator('#doc').is_visible()
    page.screenshot(path=str(out/'mobile-focused.png'))
    browser.close()
(out/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
