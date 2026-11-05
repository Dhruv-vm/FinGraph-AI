"""Regenerate frontend/src/components/home/homeData.ts from real project data.

Usage: python3 scripts/build_homepage_data.py .
Reads data/processed/documents (filing metadata) and frontend/src/lib/graphFallbackData.ts
(relationships extracted from SEC filings). Nothing is invented.
"""
import json,glob,sys,os
from collections import defaultdict,Counter
root=sys.argv[1]
docs={}
for f in glob.glob(root+'/data/processed/documents/*.json'):
    d=json.load(open(f))
    docs[d['document_id']]=dict(t=d['company'],f=d['document_type'],d=d['publication_date'])
src=open(root+'/frontend/src/lib/graphFallbackData.ts').read()
i=src.index('FALLBACK_SUBGRAPHS'); j=src.index('=',i)+1
body=src[j:].strip()
k=body.find('\nexport ')
if k>0: body=body[:k]
g=json.loads(body.strip().rstrip(';'))
TICKERS=['NVDA','MSFT','AAPL','AMZN','GOOGL','META','TSLA','JPM','XOM','JNJ']
names={}
src2=src[:i]
a=src2.index('= ',src2.index('FALLBACK_COMPANIES'))+2; comp=json.loads(src2[a:src2.rindex(']')+1])
for c in comp: names[c['ticker']]=c['canonical_name']
def edge(e):
    p=e.get('properties') or {}
    out=dict(s=e['source'],t=e['target'],r=e['relationship'],
        c=e.get('confidence'),doc=e.get('document_id'),ref=e.get('source_reference'),
        pub=e.get('publication_date') or (e.get('available_time') or '')[:10] or None,
        url=e.get('source_url'))
    a=p.get('attribution_status') or p.get('attribution')
    if a: out['a']=a
    dm=docs.get(e.get('document_id'))
    if dm: out['src']=f"{dm['t']} {dm['f']}"
    return out
showcase={}
allnodes={}
for t in TICKERS:
    sg=g[t]; cid=sg['center_node']
    nodes={n['id']:n for n in sg['nodes']}
    for n in sg['nodes']: allnodes[n['id']]=n
    direct=[e for e in sg['edges'] if cid in (e['source'],e['target']) and e['source']!=e['target']]
    # pick neighbors: round robin by relationship type, highest degree first
    byrel=defaultdict(list)
    for e in direct:
        nb=e['target'] if e['source']==cid else e['source']
        byrel[e['relationship']].append((nodes.get(nb,{}).get('degree',0),nb,e))
    for r in byrel: byrel[r].sort(key=lambda x:-x[0])
    chosen=[];seen=set();edges=[];eids=set()
    while len(chosen)<14 and any(byrel.values()):
        for r in sorted(byrel,key=lambda r:-len(byrel[r])):
            while byrel[r]:
                d,nb,e=byrel[r].pop(0)
                if nb in seen or nb not in nodes: continue
                seen.add(nb);chosen.append(nb);edges.append(e);eids.add(e['id']);break
            if len(chosen)>=14: break
    sel=set(chosen)|{cid}
    # add any other real edges among selected nodes
    for e in sg['edges']:
        if e['id'] in eids: continue
        if e['source'] in sel and e['target'] in sel and e['source']!=e['target']:
            edges.append(e);eids.add(e['id'])
    edges=edges[:26]
    showcase[t]=dict(center=cid,name=names.get(t,t),total_nodes=sg['node_count'],total_edges=sg['edge_count'],
      nodes=[dict(id=n,l=nodes[n]['label'],ty=nodes[n]['type'],deg=nodes[n]['degree']) for n in [cid]+chosen],
      edges=[edge(e) for e in edges])
# hero: top 5 per ticker
hn={};he=[];hid=set()
for t in TICKERS:
    s=showcase[t]
    for n in s['nodes'][:7]: hn[n['id']]=dict(id=n['id'],l=n['l'],c=1 if n['id']==s['center'] else 0)
for t in TICKERS:
    for e in g[t]['edges']:
        if e['source'] in hn and e['target'] in hn and e['source']!=e['target']:
            key=tuple(sorted((e['source'],e['target'])))
            if key in hid: continue
            hid.add(key);he.append([e['source'],e['target']])
filings=sorted([dict(t=v['t'],f=v['f'],d=v['d']) for v in docs.values()],key=lambda x:x['d'])
print(len(docs),Counter(v['f'] for v in docs.values()),len(hn),len(he),file=sys.stderr)
out=f"""/**
 * Homepage data — a compact, read-only extract of real project data.
 * Generated from data/processed/documents (filing metadata) and
 * src/lib/graphFallbackData.ts (relationships extracted from SEC filings).
 * No relationships are invented: every edge below exists in the source graph
 * snapshot and keeps its source filing reference.
 */

export interface HomeNode {{ id: string; l: string; ty: string; deg: number }}
export interface HomeEdge {{
  s: string; t: string; r: string; c?: number | null; doc?: string | null; ref?: string | null;
  pub?: string | null; url?: string | null; src?: string; a?: string;
}}
export interface HomeSubgraph {{ center: string; name: string; total_nodes: number; total_edges: number; nodes: HomeNode[]; edges: HomeEdge[] }}

export const HOME_TICKERS = {json.dumps(TICKERS)} as const;

export const HOME_SHOWCASE: Record<string, HomeSubgraph> = {json.dumps(showcase,separators=(',',':'))};

/** Hero network: company nodes and their highest-degree neighbours, with the real edges between them. */
export const HERO_GRAPH: {{ nodes: {{ id: string; l: string; c: number }}[]; edges: [string, string][] }} = {json.dumps(dict(nodes=list(hn.values()),edges=he),separators=(',',':'))};

/** All filings in the corpus (ticker, form type, publication date). */
export const HOME_FILINGS: {{ t: string; f: string; d: string }}[] = {json.dumps(filings,separators=(',',':'))};
"""
os.makedirs(root+'/frontend/src/components/home',exist_ok=True)
open(root+'/frontend/src/components/home/homeData.ts','w').write(out)
print(len(out),file=sys.stderr)
