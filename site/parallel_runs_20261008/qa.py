import asyncio,json,pathlib,sys
from playwright.async_api import async_playwright
ROOT=pathlib.Path('/opt/agillm43_netserve/parallel_runs_20261008')
URL='https://codex.opentransformers.online/agillm-ui'
async def main():
 result={'live':'--live' in sys.argv,'checks':{},'errors':[]}
 async with async_playwright() as p:
  browser=await p.chromium.launch(headless=True,args=['--no-sandbox'])
  context=await browser.new_context(viewport={'width':1440,'height':1000})
  page=await context.new_page()
  page.on('pageerror',lambda e:result['errors'].append(str(e)))
  if not result['live']:
   text=(ROOT/'candidate.html').read_text()
   await context.route('**/agillm-ui*',lambda route:route.fulfill(status=200,body=text,content_type='text/html'))
  response=await page.goto(URL,wait_until='domcontentloaded',timeout=45000)
  await page.wait_for_function('!!window.AGILLMParallelWorkspace',timeout=20000)
  result['checks']['page']={'status':response.status,'title':await page.title(),'url':page.url,'editor':await page.locator('#doc').count()}
  await page.locator('#doc').fill('import os\nimport sys\n')
  await page.locator('#agiAddParallel').click()
  await page.wait_for_function('window.AGILLMParallelWorkspace.state().length===2&&window.AGILLMParallelWorkspace.state()[1].model',timeout=30000)
  frame=page.frames[1]
  await frame.wait_for_function('!!window.AGILLMParallelView',timeout=15000)
  await page.wait_for_timeout(800)
  before=await page.locator('#doc').input_value()
  await frame.locator('#doc').fill('from pathlib import Path\n')
  assert await page.locator('#doc').input_value()==before
  result['checks']['isolated_editors']=True
  result['checks']['panels']=await page.evaluate('window.AGILLMParallelWorkspace.state()')
  result['checks']['main_editor_box']=await page.locator('#doc').bounding_box()
  result['checks']['child_editor_box']=await frame.locator('#doc').bounding_box()
  await page.screenshot(path=str(ROOT/'desktop-two.png'))
  await page.locator('#agiParallelMore').click()
  await page.locator('#agiParallelMore').click()
  await page.wait_for_function('window.AGILLMParallelWorkspace.state().length===4',timeout=15000)
  assert await page.locator('#agiParallelMore').is_disabled()
  result['checks']['four_panel_limit']=True
  await page.wait_for_function('window.AGILLMParallelWorkspace.state().every(x=>x.model)',timeout=40000)
  await page.locator('#agiParallelSingle').click()
  assert await page.locator('#agiParallelShell').count()==0
  assert await page.locator('#doc').input_value()==before
  result['checks']['single_view_preserves_editor']=True
  await page.set_viewport_size({'width':390,'height':844})
  await page.locator('#agiAddParallel').click()
  await page.wait_for_function('window.AGILLMParallelWorkspace.state().length===2&&window.AGILLMParallelWorkspace.state()[1].model',timeout=30000)
  await page.screenshot(path=str(ROOT/'mobile-two.png'))
  result['checks']['mobile']={'width':390,'page_width':await page.evaluate('document.documentElement.scrollWidth'),'grid_columns':await page.locator('#agiParallelGrid').evaluate('(e)=>getComputedStyle(e).gridTemplateColumns')}
  result['checks']['no_runtime_errors']=not result['errors']
  await browser.close()
 (ROOT/('qa-live.json' if result['live'] else 'qa-candidate.json')).write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2))
asyncio.run(main())
