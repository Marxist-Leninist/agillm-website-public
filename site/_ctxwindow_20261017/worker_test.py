import json, subprocess, sys, time
CMD = ["/usr/bin/python3","-u","agillm44_operator_jsonl_server.py","--checkpoint",
 "/mnt/geth-vol1/agillm44/detachable_50m/step02803693/detachable_50m_step02803693_multimode.pt",
 "--expected-sha256","5ae7741397497fe82813ab5d9e4a64017f288644c0ece97bc7f08fc086540432",
 "--expected-tokenizer-sha256","0e546c51529290ed4b18dc8b3e481a9ea39a2df2175b65ecb281af6bac79af73",
 "--device","cpu","--dtype","float32","--cpu-threads","2","--server","--worker-mode",sys.argv[1]]
env={"CUDA_VISIBLE_DEVICES":"","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","OMP_NUM_THREADS":"2","MKL_NUM_THREADS":"2","OPENBLAS_NUM_THREADS":"2","PATH":"/usr/bin:/bin"}
if sys.argv[1]=="nat": env["AGILLM_NAT_FORCE_ONE_PASS"]="1"
p=subprocess.Popen(CMD,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=env,cwd=".")
t0=time.time()
line=p.stdout.readline(); print("READY?",line.strip()[:80],"after %.1fs"%(time.time()-t0))
long_prompt=("The quick brown fox jumps over the lazy dog near the riverbank while the farmer watches quietly. ")*160+"Finally the fox"
mode={"ar":"ar","sat_fixed":"sat","sat_variable":"sat","nat":"nat"}[sys.argv[1]]
for label,prompt in (("LONG",long_prompt),("SHORT","Finally the fox")):
    req={"prompt":prompt,"mode":mode,"var":sys.argv[1]=="sat_variable","max_new":8,"min_new":0,"temperature":0.7,"top_p":0.95,"stream":True}
    if sys.argv[1]=="nat": req["nat_passes"]=1
    p.stdin.write(json.dumps(req)+"\n"); p.stdin.flush()
    t1=time.time(); out=[]
    while True:
        l=p.stdout.readline()
        if not l: break
        out.append(l.rstrip("\n"))
        if "[INFER_SERVER_RESULT_END]" in l or "[INFER_SERVER_ERROR]" in l: break
    keep=[l for l in out if l.startswith(("[prompt-window]","[INFER_SERVER_ERROR]","[STREAM_BEGIN]"))]
    toks=sum(1 for l in out if l.startswith("[STREAM_"))
    print(label,"%.1fs"%(time.time()-t1),"stream_lines",toks)
    for l in keep: print("   ",l[:300])
    print("    completion:",repr(out[-2][:120]) if len(out)>=2 else out)
p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush(); p.wait(timeout=20)
err=p.stderr.read(); print("STDERR tail:",err[-400:])
