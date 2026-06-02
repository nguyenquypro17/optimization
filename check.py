import sys

def parse_input(path):
    tok = open(path).read().split()
    N = int(tok[1]); nd = int(tok[3]); idx = 4
    t = [[0]*(N+1) for _ in range(N+1)]
    for _ in range(nd):
        a,b,d = int(tok[idx]),int(tok[idx+1]),int(tok[idx+2]); t[a][b]=d; idx+=3
    TL, TT = int(tok[idx+1]), int(tok[idx+2]); idx+=3
    m = int(tok[idx+1]); idx+=2
    dep={}
    for _ in range(m): dep[int(tok[idx])]=int(tok[idx+1]); idx+=2
    reqs={}
    while idx<len(tok) and tok[idx]=="REQ":
        rid=int(tok[idx+1]); size=int(tok[idx+2])
        reqs[rid]=dict(q=2 if size==40 else 1,a=int(tok[idx+3]),pa=tok[idx+4],pdur=int(tok[idx+5]),
                       b=int(tok[idx+6]),da=tok[idx+7],ddur=int(tok[idx+8])); idx+=9
    return dict(N=N,t=t,TL=TL,TT=TT,m=m,dep=dep,reqs=reqs)

def check(inp, outp):
    D = parse_input(inp); t=D["t"]; TL=D["TL"]; TT=D["TT"]; dep=D["dep"]; reqs=D["reqs"]
    lines=[l.strip() for l in open(outp) if l.strip()!=""]
    assert lines[0].startswith("ROUTES"), "thieu ROUTES"
    served=set(); F1=0; F2=0; i=1
    while i < len(lines):
        assert lines[i].startswith("TRUCK"), f"mong TRUCK tai {lines[i]}"
        k=int(lines[i].split()[1]); i+=1
        pos=dep[k]; tm=0; tr=0; hh=0; load=0
        # trang thai pickup cho rang buoc tra
        picked={}  # rid -> True khi da lay (de kiem tra lay truoc tra, cung truck)
        while not lines[i].endswith("STOP"):
            p=lines[i].split(); i+=1
            loc=int(p[0]); act=p[1]
            ref=int(p[2]) if len(p)>2 else None
            d=t[pos][loc]; tr+=d; tm+=d
            if act=="PICKUP_TRAILER":
                assert hh==0 and load==0 and loc==TL; hh=1; tm+=TT
            elif act=="DROP_TRAILER":
                assert hh==1 and load==0 and loc==TL; hh=0; tm+=TT
            elif act=="PICKUP_CONTAINER":
                r=reqs[ref]; assert r["pa"]=="PICKUP_CONTAINER"; assert loc==r["a"]
                assert hh==1, "PICKUP_CONTAINER ma chua co ro-mooc"
                assert load + r["q"] <= 2; load+=r["q"]; tm+=r["pdur"]; picked[ref]=True
            elif act=="PICKUP_CONTAINER_TRAILER":
                r=reqs[ref]; assert r["pa"]=="PICKUP_CONTAINER_TRAILER"; assert loc==r["a"]
                assert hh==0 and load==0, "PCT khi dang co ro-mooc"
                hh=1; load=r["q"]; tm+=r["pdur"]; picked[ref]=True
            elif act=="DROP_CONTAINER":
                r=reqs[ref]; assert r["da"]=="DROP_CONTAINER"; assert loc==r["b"]
                assert picked.get(ref), "tra truoc khi lay"
                assert hh==1 and load>=r["q"]; load-=r["q"]; tm+=r["ddur"]; served.add(ref)
            elif act=="DROP_CONTAINER_TRAILER":
                r=reqs[ref]; assert r["da"]=="DROP_CONTAINER_TRAILER"; assert loc==r["b"]
                assert picked.get(ref), "tra truoc khi lay"
                assert hh==1 and load==r["q"]; hh=0; load=0; tm+=r["ddur"]; served.add(ref)
            else:
                raise AssertionError("action la: "+act)
            pos=loc
        # dong STOP
        sp=lines[i].split(); assert int(sp[0])==dep[k] and sp[1]=="STOP", "STOP sai bai"
        i+=1
        d=t[pos][dep[k]]; tr+=d; tm+=d; pos=dep[k]
        assert hh==0 and load==0, "ket thuc van con ro-mooc/hang"
        assert lines[i]=="#", "thieu # cuoi khoi"; i+=1
        F1=max(F1,tm); F2+=tr
    missing=set(reqs.keys())-served
    assert not missing, f"chua phuc vu yeu cau: {sorted(missing)}"
    print(f"HOP LE. F1={F1}  F2={F2}  so_yeu_cau={len(reqs)}  da_phuc_vu={len(served)}")
    return F1,F2

if __name__=="__main__":
    check(sys.argv[1], sys.argv[2])
