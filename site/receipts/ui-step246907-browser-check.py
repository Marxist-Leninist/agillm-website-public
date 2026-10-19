from playwright.sync_api import sync_playwright
from pathlib import Path
import json
out=Path('/opt/agillm43_netserve/receipts')
r={'ok':False,'errors':[]}
with sync_playwright() as p:
 b=p.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage','--disable-gpu'])
 pg=b.new_page(viewport={'width':390,'height':844})
 pg.on('pageerror',lambda e:r['errors'].append(str(e)))
 try:
  pg.goto('https://codex.opentransformers.online/agillm-ui',wait_until='domcontentloaded',timeout=60000)
  pg.wait_for_selector('#servingNowStrip',timeout=10000)
  pg.wait_for_timeout(2500)
  r['serving_strip']=pg.locator('#servingNowStrip').inner_text()
  r['training_heading']=pg.locator('.training-heading').inner_text()
  r['rpv_card']=pg.locator('#agi-live .lc.rpv').inner_text()[:1200]
  r['standard_card']=pg.locator('#agi-live .lc.std').inner_text()[:1200]
  r['mobile_overflow']=pg.evaluate('document.documentElement.scrollWidth>innerWidth')
  r['mode_buttons']=pg.locator('#modeseg button:not([hidden])').all_inner_texts()
  r['ok']='246,907' in r['serving_strip'] and '2,975,219' in r['serving_strip'] and not r['mobile_overflow'] and not r['errors']
  pg.screenshot(path=str(out/'ui-step246907-mobile.png'),full_page=False)
 finally:
  b.close()
(out/'ui-step246907-browser-check.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
raise SystemExit(0 if r['ok'] else 1)
