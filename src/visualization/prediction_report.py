"""Offline HTML presentation for saved IPL model outputs.

The page embeds all forecast records; it does not fetch data, call a server,
retrain a model, or require a CDN. Native SVG player charts use the same saved
values as the PNGs. A table of source fields is available only as a drill-down.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


def write_report(payload: dict, destination: Path) -> None:
    def safe(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        return value

    encoded = json.dumps(safe(payload), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    # Prevent names or notes containing HTML from closing the JSON script tag.
    encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    destination.write_text(TEMPLATE.replace("__REPORT_DATA__", encoded), encoding="utf-8")


TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>IPL prediction report</title>
<style>
:root{--ink:#172a3a;--muted:#5c6976;--blue:#2463a5;--teal:#177965;--gold:#ad6d17;--line:#dce2e6;--paper:#f4f6f8;--white:#fff}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 system-ui,-apple-system,Segoe UI,Arial,sans-serif;color:var(--ink);background:var(--paper)}
header{background:var(--ink);color:white;padding:30px max(24px,calc((100vw - 1380px)/2)) 24px}header h1{font-size:32px;font-weight:650;margin:0 0 5px}header p{margin:0;color:#d9e1e8;max-width:1040px}
nav{display:flex;gap:6px;flex-wrap:wrap;margin:20px 0 0}button,.button{font:inherit;cursor:pointer;border:1px solid var(--line);background:white;color:var(--ink);padding:9px 14px;border-radius:7px;text-decoration:none}
nav button{border-color:#486071;background:transparent;color:white}nav button[aria-selected=true]{background:white;color:var(--ink)}button:focus-visible,select:focus-visible,input:focus-visible,a:focus-visible{outline:3px solid #d08d24;outline-offset:2px}
main{max-width:1428px;margin:auto;padding:22px 24px 50px}.panel[hidden]{display:none}.panel h2{font-size:24px;font-weight:650;margin:0 0 12px}.sub{color:var(--muted);margin:5px 0 18px}
.card{padding:20px;border:1px solid var(--line);border-radius:10px;background:white;min-width:0}.card h3{font-size:18px;margin:0 0 10px;font-weight:650}.card h4{font-size:15px;margin:16px 0 8px}.card p{margin:7px 0}.muted,small{color:var(--muted)}small{font-size:12px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.wide{grid-column:1/-1}.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:16px 0 22px}.kpi{padding:16px 18px;background:white;border:1px solid var(--line);border-radius:8px}.kpi strong{font-size:28px;display:block;font-weight:650;color:var(--ink)}.kpi span{color:var(--muted);font-size:13px}
.note{background:#edf2f7;border-left:3px solid var(--blue);padding:12px 15px;border-radius:0 6px 6px 0;margin:16px 0}.warning{background:#fff5e6;border-left-color:var(--gold)}.controls{display:flex;flex-wrap:wrap;align-items:end;gap:12px;margin:0 0 18px}.controls label{display:grid;gap:5px;min-width:170px;font-size:13px;font-weight:600}input,select{font:inherit;border:1px solid #bac5ce;border-radius:6px;padding:10px;background:white;color:var(--ink);max-width:100%}input{min-width:260px}select{min-width:200px}#playerSelect{max-width:600px}.controls .grow{flex:1}
.gallery{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.chart{border:1px solid var(--line);border-radius:10px;background:white;overflow:hidden}.chart a{display:block}.chart img{display:block;width:100%;height:auto}.chart figcaption{padding:12px 17px}.chart h3{font-size:17px;margin:0 0 5px}.chart p{font-size:13px;color:var(--muted);margin:4px 0}.chart figure{margin:0}.empty{padding:25px;color:var(--muted)}
.tag{display:inline-block;padding:3px 8px;margin:0 4px 5px 0;border:1px solid var(--line);border-radius:5px;background:#f4f6f8;font-size:12px;color:#364e61}.badge{font-weight:650}.meter{height:7px;background:#edf0f3;border-radius:6px;overflow:hidden;margin:4px 0 13px}.meter span{display:block;height:100%;background:var(--blue)}.meter.value span{background:var(--teal)}.scores{display:grid;grid-template-columns:1fr 1fr;gap:16px}.scores b{font-size:20px}.metricline{display:flex;gap:20px;flex-wrap:wrap;margin:10px 0}.metricline span{font-size:13px}.metricline strong{display:block;font-size:20px;font-weight:600}.reason{font-size:13px;color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.namebutton{border:0;padding:0;background:none;color:var(--blue);font-size:18px;text-align:left;font-weight:650}.status{font-size:12px;margin-top:8px}.status .bad{color:#934616}.status .good{color:var(--teal)}
svg{width:100%;height:auto;display:block}svg text{font-family:system-ui,Segoe UI,Arial,sans-serif;fill:#394c5d;font-size:12px}.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:12px;margin:8px 0;color:var(--muted)}.key{display:inline-block;width:12px;height:10px;margin-right:5px}.sourcewrap{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;vertical-align:top;padding:10px 12px;border-bottom:1px solid var(--line)}th{background:#edf2f7;font-weight:650}td.number{text-align:right}td a{color:var(--blue)}details{margin:14px 0}summary{cursor:pointer;font-weight:600}.pagination{display:flex;align-items:center;justify-content:space-between;gap:10px;margin:16px 0}.tabsnote{max-width:1100px}.rawtable td:first-child{width:40%;color:var(--muted)}.rawtable td{word-break:break-word}a{color:var(--blue)}
@media(max-width:1000px){.cards{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:720px){main{padding:18px 12px}header{padding:22px 18px}header h1{font-size:26px}.grid,.gallery,.cards{grid-template-columns:1fr}.kpis{grid-template-columns:repeat(2,1fr)}.controls label,.controls input,.controls select{width:100%;min-width:0}.wide{grid-column:auto}.card{padding:16px}.kpi strong{font-size:23px}}
@media print{nav,.controls,.pagination,button{display:none}header{background:white;color:#172a3a;padding:0}header p{color:#5c6976}.card,.chart{break-inside:avoid}.panel[hidden]{display:none}.gallery{grid-template-columns:1fr}main{max-width:none}.note{font-size:12px}}
</style>
</head>
<body>
<header><h1 id="reportTitle">IPL prediction report</h1><p id="reportSubtitle"></p>
<nav aria-label="Report sections">
<button data-tab="overview" aria-selected="true">Overview</button><button data-tab="players" aria-selected="false">Explore players</button><button data-tab="teams" aria-selected="false">Explore teams</button><button data-tab="charts" aria-selected="false">All charts</button><button data-tab="evaluation" aria-selected="false">Model evaluation</button><button data-tab="sources" aria-selected="false">Sources &amp; assumptions</button>
</nav></header>
<main>
<div id="warnings"></div>
<section id="overview" class="panel"><h2>What does the model expect next?</h2><p class="sub">Start with the leaders, then explore any player or team. Leader charts are selections; the searchable views include the complete saved forecast pool.</p><div id="overviewKpis" class="kpis"></div><div id="overviewCharts" class="gallery"></div><div id="overviewAssumption" class="note"></div></section>
<section id="players" class="panel" hidden><h2>Player forecasts</h2><div class="controls"><label class="grow">Search name or player ID<input id="playerSearch" placeholder="For example V Kohli, Shubman Gill, or a player ID"></label><label>Listed role<select id="playerRole"><option value="">All roles</option></select></label><label class="grow">Choose a player<select id="playerSelect"></select></label></div><p id="playerMatches" class="sub" aria-live="polite"></p><div id="playerView"></div></section>
<section id="teams" class="panel" hidden><h2>Team recommendations</h2><div class="controls"><label>Team<select id="teamSelect"></select></label><label>Player / ID<input id="teamSearch" placeholder="Search this team's complete ranking"></label><label>Role<select id="teamRole"><option value="">All roles</option></select></label><label>Scope<select id="teamScope"><option value="all">All candidates</option><option value="match">Role match and within per-player limit</option><option value="shortlist">Saved role shortlists</option></select></label></div><div id="teamIntro"></div><div id="teamCharts" class="gallery"></div><div class="pagination"><span id="teamCount" aria-live="polite"></span><div><button id="previous">Previous</button> <button id="next">Next</button></div></div><div id="teamCards" class="cards"></div></section>
<section id="charts" class="panel" hidden><h2>Charts for presentation and analysis</h2><div class="controls"><label>Topic<select id="chartTopic"><option value="">All topics</option></select></label><label>Title or team<input id="chartSearch" placeholder="Filter charts"></label></div><p class="sub">Click an image to open the full-size PNG. This gallery lists charts from this report run only, not unrelated old images left in the folder.</p><div id="allCharts" class="gallery"></div></section>
<section id="evaluation" class="panel" hidden><h2>How well did the models perform?</h2><p class="sub">These are saved chronological evaluation and selection results. They are not newly run tests and should not be presented as untouched final-test accuracy.</p><div id="metricCards" class="kpis"></div><div id="evaluationCharts" class="gallery"></div><div class="note">MAE is the average size of an absolute prediction error. Lower is better. A model that improves average error can still miss individual star seasons. Team fit and investment scores are ranking aids, not probabilities of success.</div><div id="allMetrics"></div></section>
<section id="sources" class="panel" hidden><h2>Sources, assumptions and output coverage</h2><div id="sourceAssumptions" class="card"></div><h3>Every model artifact</h3><p class="sub">Predictions are visualised; training and excluded rows are summarised as diagnostics. Saved binary models are listed, never executed by this report.</p><div class="sourcewrap"><table><thead><tr><th>Source artifact</th><th>Rows</th><th>Where it is represented</th></tr></thead><tbody id="sourceRows"></tbody></table></div></section>
<noscript><p class="note warning">Enable JavaScript to use the offline player and team explorers. The generated PNGs remain available in this folder.</p></noscript>
</main>
<script type="application/json" id="reportData">__REPORT_DATA__</script>
<script>
"use strict";
const D=JSON.parse(document.getElementById("reportData").textContent);
const $=id=>document.getElementById(id);
const e=v=>String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const valid=v=>v!==null&&v!==undefined&&v!==""&&Number.isFinite(Number(v));
const num=(v,d=0)=>valid(v)?Number(v).toLocaleString("en-IN",{minimumFractionDigits:d,maximumFractionDigits:d}):"Not available";
const money=v=>valid(v)?`₹${num(Number(v)/1e7,2)} cr`:"Not available";
const yes=v=>v===true||["true","1","yes"].includes(String(v).toLowerCase());
const clamp=v=>Math.max(0,Math.min(100,Number(v)||0));
const people=new Map(D.players.map(p=>[p.player_id,p]));
const histories=new Map();for(const row of D.history){if(!histories.has(row.player_id))histories.set(row.player_id,[]);histories.get(row.player_id).push(row);}
for(const rows of histories.values())rows.sort((a,b)=>a.season_end_year-b.season_end_year);
const teamNames=Object.keys(D.teams).sort();const roleNames=["Batter","Wicketkeeper-batter","Bowler","All-rounder","Other"];
const lower=s=>String(s??"").toLowerCase();
const options=(values,selected="")=>values.map(v=>`<option value="${e(v)}"${v===selected?" selected":""}>${e(v)}</option>`).join("");
const metric=(label,value,detail="")=>`<div class="kpi"><strong>${e(value)}</strong><span>${e(label)}</span>${detail?`<br><small>${e(detail)}</small>`:""}</div>`;
function showTab(tab){for(const panel of document.querySelectorAll(".panel"))panel.hidden=panel.id!==tab;for(const b of document.querySelectorAll("nav button"))b.setAttribute("aria-selected",String(b.dataset.tab===tab));}
for(const b of document.querySelectorAll("nav button"))b.addEventListener("click",()=>showTab(b.dataset.tab));
function chartHtml(c){return `<article class="chart"><figure><a href="${e(c.file)}" target="_blank" rel="noopener"><img src="${e(c.file)}" alt="${e(c.title)}" loading="lazy"></a><figcaption><h3>${e(c.title)}</h3><p>${e(c.note)}</p></figcaption></figure></article>`;}
function empty(text){return `<div class="card empty">${e(text)}</div>`;}
function scenario(p,m){const mid=p[`expected_${m}`],lo=p[`expected_${m}_conservative`],hi=p[`expected_${m}_upside`];return valid(lo)&&valid(hi)&&Number(lo)<=Number(mid)&&Number(mid)<=Number(hi)?[Number(lo),Number(mid),Number(hi)]:[null,Number(mid),null];}
function measures(p){return p.role_group==="Bowler"?["wickets"]:["Batter","Wicketkeeper-batter"].includes(p.role_group)?["runs"]:["runs","wickets"];}
function historySvg(p,m){
 const rows=(histories.get(p.player_id)||[]).filter(r=>valid(r[m]));const s=scenario(p,m);const W=850,H=320,L=60,R=30,T=25,B=65;
 const max=Math.max(1,...rows.map(r=>Number(r[m])),s[2]??s[1])*1.2;const maxX=rows.length+(rows.length?1:0);const count=Math.max(1,maxX+1);const step=(W-L-R)/count;const bar=Math.min(40,step*.65);const y=v=>H-B-Number(v)/max*(H-T-B);
 let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${e(p.player_name_canonical)} ${e(m)} actual IPL history and ${D.year} forecast">`;
 for(let i=0;i<=4;i++){const v=max*i/4;svg+=`<line x1="${L}" y1="${y(v)}" x2="${W-R}" y2="${y(v)}" stroke="#e5e9ed"/><text x="${L-8}" y="${y(v)+4}" text-anchor="end">${num(v,m==="wickets"?1:0)}</text>`;}
 for(let i=0;i<rows.length;i++){const r=rows[i],x=L+step*(i+.5);svg+=`<rect x="${x-bar/2}" y="${y(r[m])}" width="${bar}" height="${H-B-y(r[m])}" fill="#bdc9d7"><title>${e(r.season_end_year)} actual ${e(m)}: ${num(r[m],m==="wickets"?1:0)}</title></rect><text x="${x}" y="${H-B+22}" text-anchor="middle" transform="rotate(-35 ${x} ${H-B+22})">${e(r.season_end_year)}</text>`;}
 const x=L+step*(maxX+.5);const colour=m==="wickets"?"#177965":"#2463a5";
 svg+=`<rect x="${x-bar/2}" y="${y(s[1])}" width="${bar}" height="${H-B-y(s[1])}" fill="${colour}"><title>Expected ${e(m)}: ${num(s[1],m==="wickets"?1:0)}</title></rect>`;
 if(s[0]!==null)svg+=`<path d="M ${x} ${y(s[0])} V ${y(s[2])} M ${x-6} ${y(s[0])} H ${x+6} M ${x-6} ${y(s[2])} H ${x+6}" stroke="#ad6d17" stroke-width="2" fill="none"><title>Conservative ${num(s[0],1)}; upside ${num(s[2],1)}. Scenario, not a confidence interval.</title></path>`;
 svg+=`<text x="${x}" y="${H-B+21}" text-anchor="middle">${D.year}</text><text x="${x}" y="${H-B+40}" text-anchor="middle">Forecast</text></svg>`;return svg;
}
function acrossTeamsSvg(p){
 const rows=teamNames.map(team=>({team,row:D.teams[team].find(r=>r.player_id===p.player_id)})).filter(r=>r.row);
 if(!rows.length)return empty("No team-fit outputs are available for this player.");
 const W=1080,L=270,R=355,step=46,H=rows.length*step+65,plotWidth=W-L-R;
 let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Fit before the hard budget gate and budget status for ${e(p.player_name_canonical)}">`;
 s+=`<text x="${L}" y="17">Fit before the hard price-limit block</text><text x="${W-R+20}" y="17">Price-limit status</text>`;
 for(const threshold of [0,40,60,100]){const x=L+plotWidth*threshold/100;s+=`<line x1="${x}" y1="26" x2="${x}" y2="${H-27}" stroke="#d8dfe5" stroke-dasharray="4 4"/><text x="${x}" y="${H-7}" text-anchor="middle">${threshold}</text>`;}
 rows.forEach(({team,row},i)=>{
  const y=i*step+35,hasBudget=Object.prototype.hasOwnProperty.call(row,"within_budget")&&row.within_budget!==null;
  const blocked=hasBudget&&!yes(row.within_budget),hasRaw=valid(row.pre_budget_fit_score);
  // A legacy budget-blocked zero does not reveal potential fit. Do not invent
  // a score by reversing the gate or substitute an investment score for it.
  const score=hasRaw?Number(row.pre_budget_fit_score):blocked?null:valid(row.team_fit_score)?Number(row.team_fit_score):null;
  const colour=blocked?"#ad6d17":"#2463a5",width=score===null?0:plotWidth*clamp(score)/100;
  s+=`<text x="${L-12}" y="${y+15}" text-anchor="end">${e(team)}</text><rect x="${L}" y="${y}" width="${plotWidth}" height="22" rx="3" fill="#f0f3f6"/>`;
  if(score!==null){const inside=score>=30;s+=`<rect x="${L}" y="${y}" width="${width}" height="22" rx="3" fill="${colour}"/><text x="${inside?L+width-6:L+width+5}" y="${y+15}" text-anchor="${inside?"end":"start"}" style="fill:${inside?"white":"#394c5d"}">${num(score,1)}</text>`;}
  else s+=`<text x="${L+6}" y="${y+15}">Pre-budget fit not recorded</text>`;
  const status=blocked?"Over per-player limit":hasBudget?"Within per-player limit":"Budget check unavailable";
  const limit=valid(row.max_fair_value_inr)?`Limit ${money(row.max_fair_value_inr)}`:"";
  const final=valid(row.team_fit_score)?`Final score ${num(row.team_fit_score,1)}`:"";
  s+=`<text x="${W-R+20}" y="${y+13}" style="fill:${colour}">${e(status)}</text><text x="${W-R+20}" y="${y+29}" style="font-size:10px">${e([limit,final].filter(Boolean).join("; "))}</text>`;
 });
 return s+`</svg><p class="muted">Estimated player value: ${e(money(p.predicted_fair_value_inr))}. Amber bars mean outside the stated per-player limit, not zero cricketing suitability. The pre-budget score still includes the existing role, performance and value weights. Final budget-gated rankings remain unchanged.</p>`;
}
function rawDetails(p){return `<details><summary>Inspect all saved player output fields</summary><div class="sourcewrap"><table class="rawtable"><tbody>${Object.entries(p).map(([k,v])=>`<tr><td>${e(k)}</td><td>${e(v===null||v===undefined?"Not available":typeof v==="object"?JSON.stringify(v):v)}</td></tr>`).join("")}</tbody></table></div></details>`;}
function renderPlayer(){const p=people.get($("playerSelect").value);if(!p){$("playerView").innerHTML=empty("No players match that name or role. Try another search or use the exact player ID.");return;}let cards="";let trends="";
 for(const m of measures(p)){const d=m==="runs"?0:1,s=scenario(p,m),appearance=m==="runs"?"batting":"bowling",matches=p[`expected_${appearance}_matches`],rate=valid(matches)&&Number(matches)>0?s[1]/Number(matches):null;cards+=metric(`Expected ${D.year} ${m}`,num(s[1],d))+metric(`Expected ${appearance} appearances`,num(matches,1));trends+=`<article class="card"><h3>${m==="runs"?"Batting output":"Bowling output"}</h3>${historySvg(p,m)}<div class="legend"><span><i class="key" style="background:#bdc9d7"></i>Actual IPL seasons</span><span><i class="key" style="background:${m==="runs"?"#2463a5":"#177965"}"></i>Expected forecast</span><span>Whisker: supplied scenario range</span></div><p>Conservative: <strong>${num(s[0],d)}</strong> · Expected: <strong>${num(s[1],d)}</strong> · Upside: <strong>${num(s[2],d)}</strong></p><p class="muted">Implied ${e(m)} per ${appearance} appearance: ${num(rate,2)}. ${s[0]===null?"Scenario ranges were not supplied by this saved forecast.":"The range is a scenario, not a statistical confidence interval."}</p></article>`;}
 cards+=metric("Estimated fair value",money(p.predicted_fair_value_inr));if(valid(p.investment_score))cards+=metric("Investment score / 100",num(p.investment_score,1),p.investment_label||"");
 const evidence=[["has_recent_ipl_history","IPL"],["has_recent_t20_history","Other T20"],["has_recent_odi_history","ODI"]].map(([k,label])=>`<span class="tag">${label}: ${yes(p[k])?"recent evidence":"not available"}</span>`).join("");
 const countryNote=p.country_used_for_screen?`<br>Country used for the pool screen: <strong>${e(p.country_used_for_screen)}</strong>. ${e(p.country_evidence_basis||"")}. Rule basis: ${e(p.country_rule_basis||"not configured")}.`:"";
 const eligibility=valid(p.recent_qualifying_matches)?`<div class="note"><strong>Candidate-pool screen:</strong> ${e(p.activity_window_start)} to ${e(p.activity_window_end)}. Recorded IPL matches: ${num(p.recent_ipl_matches)}; supported international matches: ${num(p.recent_international_matches)}.<br>Next-season availability: <strong>${e(p.availability_status||"unknown")}</strong>. ${e(p.availability_evidence||"")}${countryNote}<br>${e(p.fitness_status||"Medical fitness is not assessed by this model.")}</div>`:"";
 const medians=[];for(const m of measures(p)){const v=p[`historical_median_ipl_${m}`];if(valid(v))medians.push(`Trailing IPL median ${m}: ${num(v,m==="runs"?0:1)}`);}
 $("playerView").innerHTML=`<div class="card"><h2>${e(p.player_name_canonical)}</h2><span class="tag">${e(p.role_group)}</span><span class="tag">Player ID ${e(p.player_id)}</span><span class="tag">${e(p.nationality||"Nationality unavailable")}</span><p class="muted">${e(p.metadata_playing_role||p.role||"Role metadata unavailable")}. Forecast method: ${e(p.forecast_method||"not recorded in this version")}.</p><div>${evidence}</div>${medians.length?`<p class="muted">${e(medians.join("; "))}</p>`:""}</div>${eligibility}<div class="kpis">${cards}</div><div class="grid">${trends}<article class="card wide"><h3>How does this player fit each team?</h3>${acrossTeamsSvg(p)}<p class="muted">Fit is measured against each team's chosen requirements. >60 Strong fit; >40 to 60 Consider; ≤40 Low fit. It is not a percentage chance of success.</p></article></div><div class="note">The forecast assumes this player appears in IPL. It does not predict retention, release, registration, injury status, or selection. Strike rate, economy and wicketkeeping dismissals are not forecast by the saved model.</div>${rawDetails(p)}`;
}
function filterPlayers(){const term=lower($("playerSearch").value).trim(),role=$("playerRole").value,old=$("playerSelect").value;const list=D.players.filter(p=>(!role||p.role_group===role)&&(!term||lower(p.player_name_canonical).includes(term)||lower(p.player_id).includes(term))).sort((a,b)=>String(a.player_name_canonical).localeCompare(String(b.player_name_canonical)));$("playerSelect").innerHTML=list.map(p=>`<option value="${e(p.player_id)}">${e(p.player_name_canonical)} — ${e(p.role_group)} (${e(p.player_id)})</option>`).join("");if(list.some(p=>p.player_id===old))$("playerSelect").value=old;$("playerMatches").textContent=`${list.length.toLocaleString()} matching players out of ${D.players.length.toLocaleString()} saved forecasts. Names are joined by player ID, never guessed from abbreviations.`;renderPlayer();}
function openPlayer(id){$("playerSearch").value="";$("playerRole").value="";filterPlayers();$("playerSelect").value=id;renderPlayer();showTab("players");window.scrollTo({top:0,behavior:"smooth"});}
let page=0;const PAGE_SIZE=12;
function renderTeam(reset=false){if(reset)page=0;const team=$("teamSelect").value,ranking=D.teams[team]||[],role=$("teamRole").value,term=lower($("teamSearch").value).trim(),scope=$("teamScope").value;const shortlistIds=new Set((D.shortlists[team]||[]).map(r=>r.player_id));const merged=ranking.map(r=>({...people.get(r.player_id),...r}));const filtered=merged.filter(p=>(!role||p.role_group===role)&&(!term||lower(p.player_name_canonical).includes(term)||lower(p.player_id).includes(term))&&(scope!=="match"||(yes(p.role_match)&&yes(p.within_budget)))&&(scope!=="shortlist"||shortlistIds.has(p.player_id)));const pages=Math.max(1,Math.ceil(filtered.length/PAGE_SIZE));page=Math.min(page,pages-1);$("teamIntro").innerHTML=team?`<p class="sub">${e(team)}: ${ranking.length.toLocaleString()} ranked players. Fit and investment answer different questions. The budget check is a per-player price limit, not the team's remaining purse.</p>`:empty("No team rankings are available. Generate team-fit outputs to enable this view.");$("teamCharts").innerHTML=D.charts.filter(c=>c.team===team).map(chartHtml).join("");$("teamCount").textContent=`${filtered.length.toLocaleString()} matching players · page ${page+1} of ${pages}`;$("previous").disabled=page===0;$("next").disabled=page>=pages-1;
 $("teamCards").innerHTML=filtered.slice(page*PAGE_SIZE,(page+1)*PAGE_SIZE).map(p=>`<article class="card"><button class="namebutton" data-player-id="${e(p.player_id)}">${e(p.player_name_canonical)}</button><p><span class="tag">${e(p.role_group)}</span><span class="tag">Rank ${num(p.rank)}</span></p><div class="scores"><div>Team fit <b>${num(p.team_fit_score,1)}</b><div class="meter"><span style="width:${clamp(p.team_fit_score)}%"></span></div></div><div>Investment <b>${num(p.investment_score,1)}</b><div class="meter value"><span style="width:${clamp(p.investment_score)}%"></span></div></div></div><div class="metricline"><span>Estimated fair value<strong>${e(money(p.predicted_fair_value_inr))}</strong></span>${measures(p).map(m=>`<span>Expected ${e(m)}<strong>${num(p[`expected_${m}`],m==="runs"?0:1)}</strong></span>`).join("")}</div><p class="badge">${e(p.recommendation)}; ${e(p.investment_label||"investment label unavailable")}</p><p class="status"><span class="${yes(p.role_match)?"good":"bad"}">${yes(p.role_match)?"Role match":"Role mismatch"}</span> · <span class="${yes(p.within_budget)?"good":"bad"}">${yes(p.within_budget)?"Within per-player limit":"Over per-player limit"}</span></p><details><summary>Why this score?</summary><p class="reason">${e(p.reason||"No saved explanation is available.")}</p>${rawDetails(p)}</details></article>`).join("")||empty("No players match these team filters.");
 for(const b of $("teamCards").querySelectorAll("[data-player-id]"))b.addEventListener("click",()=>openPlayer(b.dataset.playerId));
}
function renderGallery(){const topic=$("chartTopic").value,term=lower($("chartSearch").value);$("allCharts").innerHTML=D.charts.filter(c=>(!topic||c.section===topic)&&(!term||lower(c.title+" "+c.team).includes(term))).map(chartHtml).join("")||empty("No charts match the filter.");}
$("reportTitle").textContent=`IPL ${D.year}: player and auction forecasts`;
$("reportSubtitle").textContent=`${D.players.length.toLocaleString()} forecast players · ${teamNames.length} team profiles · inputs through ${D.asOf}. A visual view of the saved model results; no forecasts are changed here.`;
if(D.warnings.length)$("warnings").innerHTML=`<details class="note warning"><summary>${D.warnings.length} source-availability notes</summary>${D.warnings.map(w=>`<p>${e(w)}</p>`).join("")}</details>`;
$("overviewKpis").innerHTML=metric("Forecast players",num(D.players.length))+metric("Team profiles",num(teamNames.length))+metric("Forecast season",String(D.year))+metric("Generated presentation charts",num(D.charts.length));
const overviewFiles=["predicted_runs_leaders.png","predicted_wickets_leaders.png","forecast_pool.png","investment_leaders.png"];
$("overviewCharts").innerHTML=D.charts.filter(c=>overviewFiles.includes(c.file)).map(chartHtml).join("");$("overviewAssumption").textContent=D.assumption;
$("playerRole").insertAdjacentHTML("beforeend",options(roleNames));$("teamRole").insertAdjacentHTML("beforeend",options(roleNames));$("teamSelect").innerHTML=options(teamNames);
$("chartTopic").insertAdjacentHTML("beforeend",options([...new Set(D.charts.map(c=>c.section))].sort()));
$("sourceAssumptions").innerHTML=`<p><strong>Forecast season:</strong> ${D.year}. <strong>Input cutoff:</strong> ${D.asOf}. <strong>Report generated:</strong> ${e(D.generated)}.</p><p>${e(D.assumption)}</p><p>Conservative and upside are the saved model scenarios, not confidence limits. When a field is absent, it is shown as unavailable, never replaced with a fabricated zero.</p><p>Batting and bowling appearances are model outputs, not selection probabilities. Investment is a 0–100 relative performance-for-price score, not financial ROI. Team-fit scores use your team-needs assumptions; they do not enforce roster composition, overseas-player rules or a combined squad purse.</p><p>Forecast runs/wickets, appearances, fair prices, model comparisons, evaluation errors, feature importance, training eligibility and excluded data all have visual or diagnostic views. Binary joblib files are runtime artifacts rather than presentation data.</p>`;
$("sourceRows").innerHTML=D.coverage.map(s=>`<tr><td><a href="${e(s.href)}">${e(s.file)}</a></td><td class="number">${s.rows===null?"—":num(s.rows)}</td><td>${e(s.view)}</td></tr>`).join("");
const pm=D.metrics["performance_metrics.json"]||{},fm=D.metrics["fair_value_metrics.json"]||{};let mc="";
if(valid(pm.targets?.next_runs?.mae))mc+=metric("Runs mean absolute error",num(pm.targets.next_runs.mae,1));if(valid(pm.targets?.next_wickets?.mae))mc+=metric("Wickets mean absolute error",num(pm.targets.next_wickets.mae,1));if(valid(fm.price_inr?.mae))mc+=metric("Price mean absolute error",money(fm.price_inr.mae));
$("metricCards").innerHTML=mc;$("evaluationCharts").innerHTML=D.charts.filter(c=>["Evaluation","Diagnostics"].includes(c.section)).map(chartHtml).join("");
function objectRows(value,prefix=""){return Object.entries(value).flatMap(([key,v])=>{const label=prefix?prefix+" / "+key:key;return v!==null&&typeof v==="object"&&!Array.isArray(v)?objectRows(v,label):`<tr><td>${e(label.replaceAll("_"," "))}</td><td>${e(v===null?"Not available":Array.isArray(v)?JSON.stringify(v):v)}</td></tr>`;}).join("");}
$("allMetrics").innerHTML=Object.entries({...D.metrics,"Analysis configuration":D.configuration}).map(([name,obj])=>`<details class="card"><summary>${e(name.replaceAll("_"," "))}: all recorded metrics and settings</summary><div class="sourcewrap"><table class="rawtable"><tbody>${objectRows(obj)}</tbody></table></div></details>`).join("");
$("playerSearch").addEventListener("input",filterPlayers);$("playerRole").addEventListener("change",filterPlayers);$("playerSelect").addEventListener("change",renderPlayer);
for(const id of ["teamSelect","teamRole","teamScope"])$(id).addEventListener("change",()=>renderTeam(true));$("teamSearch").addEventListener("input",()=>renderTeam(true));$("previous").addEventListener("click",()=>{page--;renderTeam();});$("next").addEventListener("click",()=>{page++;renderTeam();});$("chartTopic").addEventListener("change",renderGallery);$("chartSearch").addEventListener("input",renderGallery);
filterPlayers();const highest=D.players.reduce((a,b)=>Number(a.expected_runs)>Number(b.expected_runs)?a:b,D.players[0]);if(highest){$("playerSelect").value=highest.player_id;renderPlayer();}renderTeam();renderGallery();
</script></body></html>'''
