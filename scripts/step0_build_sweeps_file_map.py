import json,os,glob,ast
SP='/tmp/claude-256458/-mnt-lts4-scratch-home-yqin-DllmAR/3875e4fd-07a7-4a02-b1eb-7f3f045f7f36/scratchpad'
DS='/mnt/lts4/scratch/home/yqin/DllmAR/dream_study'
NM='/mnt/lts4/scratch/home/abdelrah/ARiFF/TaskVectorDLLM/step5_cross_arch/outputs'
EPO=f'{DS}/dream_coder_official_eval/instruct/code_eval/evalplus/evalplus_official/humaneval'
CK='mnt--lts4--scratch--home--yqin--DllmAR--dream_study--step2_instruct--outputs--checkpoints--'
CF=f'{DS}/step2_instruct/outputs/official_eval/coding_full'
CM=f'{DS}/step2_instruct/outputs/official_eval/coding_community/mbpp'
S8=f'{DS}/step8_rl_vector/outputs/coding'
AR8=f'{DS}/step8_rl_vector/outputs/coding_ar'
M5=f'{DS}/step7_reasoning_merge/outputs/math500'
def ep(p,plus=False):
    ev=json.load(open(p))['eval']; n=len(ev)
    ok=[v[0]['base_status']=='pass' and (not plus or v[0].get('plus_status')=='pass') for v in ev.values()]
    return n,100*sum(ok)/n
def rs(d):
    pp=json.load(open(f'{d}/results.json'))['per_problem']; return len(pp),100*sum(bool(p['ok']) for p in pp)/len(pp)
def gf(x,path):
    for k in path.split('.'):
        if isinstance(x,str): x=ast.literal_eval(x)
        x=x[k]
    return x
NF={'ifeval':('strict_eval.follow_all_instructions','key'),'math-500':('symbolic_correct','unique_id'),
    'human-eval':('is_correct','task_id'),'mbpp':('is_correct','task_id'),'gsm8k':('symbolic_correct','problem')}
def nm(d,b):
    rows=[json.loads(l) for l in open(f'{d}/eval/eval-results/{b}/output-rs0.jsonl')]
    f,_=NF[b]; return len(rows),100*sum(bool(gf(r,f)) for r in rows)/len(rows)
PK=('temperature','top_p','tokens_to_generate','steps','block_length','thinking_enabled','max_thinking_tokens','generation_algorithm','engine')
def nproto(d):
    p={}
    for l in open(f'{d}/resolved_protocol.env'):
        if '=' in l:
            k,v=l.strip().split('=',1)
            if k in PK or 'threshold' in k: p[k]=v
    return p
out={}; notes=[]
def add(key,loader,parent,ar,ar_loader,n,merge,addition,scorer,parent_pool):
    pn,pacc=scorer(parent)
    assert pn==n,(key,pn)
    S={'loader':loader,'parent':parent,'parent_acc':round(pacc,2),'parent_pool':parent_pool,'ar':ar,'ar_loader':ar_loader,'n':n,'merge':[],'addition':[]}
    for mode,pts in (('merge',merge),('addition',addition)):
        for beta,path,pool,doc in pts:
            nn,acc=scorer(path)
            if nn!=n: notes.append(f'EXCLUDED n={nn}: {key} {mode} {beta} {path}'); continue
            e={'beta':beta,'path':path,'acc':round(acc,2),'pool':pool}
            if doc is not None:
                e['doc_acc']=doc
                if abs(doc-acc)>0.06: notes.append(f'MISMATCH {key} {mode} {beta}: file {acc:.2f} vs doc {doc}')
            S[mode].append(e)
        S[mode].sort(key=lambda e:e['beta'])
    out[key]=S
# ---------------- Dream-Coder HumanEval
ef=lambda a,b: f'{EPO}/{CK}dreamcoderbase_plus_{a}vdreamcoder_{b}vqwencoder_dllm_temp_0.1_len_512.eval_results.json'
cf=lambda t: sorted(glob.glob(f'{CF}/{t}/humaneval_eval_results_*.json'))[-1]
HOP='Hopper (H100/H200, Jun-Jul evalplus_official)'
add('Dream-Coder · HumanEval','evalplus',f'{EPO}/Dream-org--Dream-Coder-v0-Instruct-7B_dllm_temp_0.1_len_512.eval_results.json',
    f'{AR8}/qwencoder-inst-tok2048/humaneval/samples-sanitized/eval_results.json','evalplus',164,
    [(0.1,ef('0.9','0.1'),HOP,None),(0.2,ef('0.8','0.2'),HOP,84.8),(0.25,ef('0.75','0.25'),HOP,None),(0.3,ef('0.7','0.3'),HOP,None),(0.4,ef('0.6','0.4'),HOP,None)],
    [(0.2,cf('dcoder-10aD-02aA'),'H100',80.5),(0.3,cf('dcoder-10aD-03aA'),'H100',82.9),(0.4,cf('dcoder-10aD-04aA'),'H100',83.5),(0.5,cf('dcoder-10aD-05aA'),'H100',82.9),(0.6,cf('dcoder-10aD-06aA'),'H100',None)],
    lambda p: ep(p),HOP)
# ---------------- Dream-Coder MBPP
mm=lambda t: f'{CM}/{t}/mbpp/eval_results.json'
add('Dream-Coder · MBPP','evalplus',mm('parent_full'),f'{AR8}/qwencoder-inst-tok2048/mbpp/samples-sanitized/eval_results.json','evalplus',378,
    [(0.05,mm('merge9505_full'),'Hopper (Aug 2)',None),(0.1,mm('merge9010_full'),'Hopper (Aug 2)',None),(0.15,mm('merge8515_full'),'Hopper (Aug 1)',None),(0.2,mm('merge8020_full'),'Hopper (Aug 2)',80.4),(0.25,mm('merge_full'),'Hopper (Jul 30)',None)],
    [(0.4,mm('custom-dcoder-10aD-04aA_full'),'H100 (Sep 14)',81.7)],
    lambda p: ep(p),'Hopper (Jul 30)')
# ---------------- DiffuCoder-SFT HumanEval
he=lambda t: f'{S8}/{t}/humaneval_T0.2_eval_results.json'; mb=lambda t: f'{S8}/{t}/mbpp_T0.2_eval_results.json'
add('DiffuCoder-SFT · HumanEval','evalplus',he('inst-t02'),f'{AR8}/qwencoder-inst-tok2048/humaneval/samples-sanitized/eval_results.json','evalplus',164,
    [(0.05,he('cvx-b005-he-t02'),'H100',72.6),(0.1,he('cvx-b010-he-t02'),'H100',76.2),(0.2,he('cvx-b020-he-t02'),'H100',79.3),(0.25,he('cvx-b025-he-t02'),'H100',73.2),(0.3,he('cvx-b030-he-t02'),'H100',77.4),(0.4,he('cvx-b040-he-t02'),'H100',78.7)],
    [(0.15,he('cvx-addonly-b015'),'H100',None),(0.2,he('cvx-addonly-a020'),'H100',72.0),(0.25,he('cvx-addonly-a025'),'H100',73.8),(0.35,he('cvx-addonly-a035'),'H100',73.2),(0.4,he('cvx-addonly-a040'),'H100',72.6),(0.5,he('cvx-addonly-a050'),'H100',73.8),(0.6,he('cvx-addonly-a060'),'H100',73.8),(0.7,he('sft-dadd-b07-he-t02'),'H100',65.9)],
    lambda p: ep(p),'H100')
add('DiffuCoder-SFT · MBPP','evalplus',mb('inst-mbpp-t02'),f'{AR8}/qwencoder-inst-tok2048/mbpp/samples-sanitized/eval_results.json','evalplus',378,
    [(0.05,mb('cvx-b005-mbpp-t02'),'H100',None),(0.1,mb('cvx-b010-mbpp-t02'),'H100',None),(0.2,mb('cvx-b020-mbpp-t02'),'H100',78.6),(0.25,mb('cvx-b025-mbpp-t02'),'H100',None),(0.3,mb('cvx-b030'),'H100',None)],
    [(0.15,mb('cvx-addonly-b015'),'H100',None),(0.2,mb('cvx-addonly-a020'),'H100',None),(0.25,mb('cvx-addonly-a025'),'H100',None),(0.35,mb('cvx-addonly-a035'),'H100',None),(0.4,mb('cvx-addonly-a040'),'H100',None),(0.6,mb('cvx-addonly-a060'),'H100',None)],
    lambda p: ep(p),'H100')
# ---------------- DiffuCoder-RL MBPP (plus)
add('DiffuCoder-RL · MBPP+','evalplus_plus',mb('grpo-mbpp-t02'),f'{AR8}/acecoder-rule-tok2048/mbpp/samples-sanitized/eval_results.json','evalplus_plus',378,
    [(0.2,mb('rlrl1-b020-mbpp-t02'),'H100',None),(0.3,mb('rlrl1-b030-mbpp-t02'),'H100',69.0),(0.4,mb('rlrl1-b040-mbpp-t02'),'H100',68.3),(0.5,mb('rlrl1-b050-mbpp-t02'),'H100',69.0),(0.6,mb('rlrl1-b060-mbpp-t02'),'H100',None)],
    [(0.4,mb('rl-dadd-b04-mbpp-t02'),'H100',66.9),(0.5,mb('rl-dadd-b05-mbpp-t02'),'H100',66.9),(0.6,mb('rl-dadd-b06-mbpp-t02'),'H100',66.4),(0.8,mb('rl-dadd-b08-mbpp-t02'),'H100',66.9)],
    lambda p: ep(p,True),'H100')
# ---------------- DreamReasoner
AR3=f'{M5}/qwen3-m500-tok8192/results.json'
add('DreamReasoner · MATH-500 @8k','reasoner',f'{M5}/parent-m500-8k/results.json',AR3,'reasoner_ar_perq_frac>=0.5',500,
    [(b/10,f'{M5}/rmg-b0{b}0-m500-8k/results.json','B300',{2:94.8}.get(b)) for b in (1,2,3,4)],
    [(b/10,f'{M5}/rmg-dadd-b0{b}-m500-8k-b300/results.json','B300',None) for b in (2,3,4,5,6,7)],
    lambda p: rs(os.path.dirname(p)),'B300')
add('DreamReasoner · MATH-500 @4k','reasoner',f'{M5}/parent/results.json',AR3,'reasoner_ar_perq_frac>=0.5',500,
    [(b/10,f'{M5}/rmg-b0{b}0/results.json','H100',{2:91.6}.get(b)) for b in (1,2,3,4)],
    [(b/10,f'{M5}/rmg-dadd-b0{b}/results.json','H100',{1:89.8,2:90.4,3:91.4,4:90.2}[b]) for b in (1,2,3,4)],
    lambda p: rs(os.path.dirname(p)),'H100')
# ---------------- Nemotron
TP=f'{NM}/famC_tokenizer_protected_sweeps'
def nemo_series(size,b,parent,ar,merge,addition,docs):
    f,k=NF[b]; ld=f"nemo_{f}_{k}"
    pp=nproto(parent)
    def sc(d): return nm(d,b)
    n=sc(parent)[0]
    for mode,pts in (('merge',merge),('addition',addition)):
        for beta,d,*_ in pts:
            q=nproto(d)
            diff={x:(pp.get(x),q.get(x)) for x in set(pp)|set(q) if pp.get(x)!=q.get(x)}
            if diff: notes.append(f'PROTO DIFF {size} {b} {mode} {beta}: {diff}')
    add(f'Nemotron-{size} · {b}',ld,parent,ar,ld,n,
        [(c,d,'H200 (Amel; not logged per dir)',docs.get(('merge',c))) for c,d in merge],
        [(c,d,'H200 (Amel; not logged per dir)',docs.get(('addition',c))) for c,d in addition],sc,'H200 (Amel; not logged per dir)')
def sumdocs(size):
    D={}
    for fn,mode in (('merge-dlm-final.json','merge'),('merge-dlm-selected.json','merge'),('addition-dlm-fill-full.json','addition')):
        p=f'{TP}/{size}/summaries/{fn}'
        if os.path.exists(p):
            for r in json.load(open(p))['results']: D[(mode,r['coefficient'])]=round(r['prompt_strict_accuracy'],2)
    return D
# 3B
ev3=f'{TP}/3b/evaluations'; dn3=f'{TP}/3b/downstream_evaluations'; AR3B=f'{NM}/famC_ar_donor_ns_native_sampled_pixi_v1/3b/seed0/sm_103a'
PR3=f'{NM}/famC_paper_reproduction/evaluations/famc-paper-repro-parent-dlm-'
nemo_series('3B','ifeval',PR3+'ifeval',f'{AR3B}/ifeval',
    [(c/10,f'{ev3}/famc-tokenizer-protected-3b-merge-dlm-ifeval-coef0{c}0') for c in (2,3,4,5,6,8)],
    [(c/10,f'{ev3}/famc-tokenizer-protected-3b-ar-addition-dlm-ifeval-coef0{c}0') for c in range(1,10)],sumdocs('3b'))
for b,mc,ac,par in (('math-500',(3,4),(4,),PR3+'math-500'),('human-eval',(3,4),(4,),PR3+'human-eval'),
                    ('mbpp',(3,),(4,),f'{NM}/famC_paper_extension/evaluations/famc-paper-ext-v1-parent-dlm-mbpp'),('gsm8k',(3,4),(),PR3+'gsm8k')):
    nemo_series('3B',b,par,f'{AR3B}/{b}',
        [(c/10,f'{dn3}/famc-tokenizer-protected-3b-merge-coef0{c}0-dlm-{b}') for c in mc],
        [(c/10,f'{dn3}/famc-tokenizer-protected-3b-ar-addition-coef0{c}0-dlm-{b}') for c in ac],{})
# 8B
ev8=f'{TP}/8b/evaluations'; dn8=f'{TP}/8b/downstream_evaluations'; AR8B=f'{NM}/famC_ar_donor_ns_native_sampled_pixi_v1/8b/seed0/sm_103a'
PR8=f'{NM}/famC_8b_paper_reproduction/evaluations/famc-8b-official-public-v1-dlm-'
TPE8=f'{NM}/famC_8b_paper_reproduction/tokenizer_protected_evaluations/famc-8b-merge-beta040-frozen-head-tokenizer-protected-v1-official-public-v1-dlm-'
nemo_series('8B','ifeval',PR8+'ifeval',f'{AR8B}/ifeval',
    [(c/10,f'{ev8}/famc-tokenizer-protected-8b-merge-dlm-ifeval-coef0{c}0') for c in (2,3,4,5)],
    [(c/10,f'{ev8}/famc-tokenizer-protected-8b-ar-addition-dlm-ifeval-coef0{c}0') for c in range(1,10)],sumdocs('8b'))
for b in ('math-500','human-eval','mbpp','gsm8k'):
    nemo_series('8B',b,PR8+b,f'{AR8B}/{b}',
        [(0.2,f'{dn8}/famc-tokenizer-protected-8b-merge-dlm-{b}-coef020'),(0.3,f'{dn8}/famc-tokenizer-protected-8b-merge-dlm-{b}-coef030'),(0.4,TPE8+b)],[],{})
json.dump(out,open(f'{SP}/sweeps.json','w'),indent=1)
for k,v in out.items():
    print(f"{k} | n={v['n']} parent={v['parent_acc']} | M: "+' '.join(f"{e['beta']}:{e['acc']}" for e in v['merge'])+" | A: "+' '.join(f"{e['beta']}:{e['acc']}" for e in v['addition']))
print('\n'.join(notes))
