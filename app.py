from __future__ import annotations

import traceback
import gradio as gr
from inference_engine import DefectInferenceEngine

ENGINE = None
ENGINE_ERROR = ""
try:
    ENGINE = DefectInferenceEngine()
except Exception as exc:
    ENGINE_ERROR = f"{type(exc).__name__}: {exc}"
    traceback.print_exc()

I18N = {
    "English": {"title":"Final V3 | Text-Guided Photovoltaic Defect Segmentation", "input":"## Input", "result":"## Detection Result", "free":"Free-text Expression", "free_info":"Text-described conditions are locked; dropdowns only supplement missing conditions.", "structured":"### Structured Query", "defect":"Defect Type", "position":"Position / Region", "pos_info":"Position filtering depends on the semantic capability learned by the current model.", "size":"Size", "relation":"Busbar Relation", "final":"Base Model Query", "manual":"Optional Manual Ground Truth Mask", "gt_note":"Automatic GT needs the original dataset image and an exact dataset expression. Manual GT always takes priority.", "start":"Start Detection", "clear":"Clear", "gt":"Ground Truth Mask", "pred":"Predicted Mask", "compare":"GT vs Prediction", "overlay":"Prediction Overlay", "metrics":"### Evaluation Metrics", "info":"## Inference Information", "model":"Model Information", "download":"Download Results", "locked":"Locked by text", "detected":"Detected from text", "supplemented":"Supplemented manually", "none":"Not specified", "note":"The model query remains English because bert-base-uncased was trained with English text."},
    "中文": {"title":"文本引导的光伏缺陷分割系统", "input":"## 输入", "result":"## 检测结果", "free":"自由文本描述", "free_info":"文本中明确描述的条件会锁定；下拉框仅补充文本未提及的条件。", "structured":"### 结构化补充条件", "defect":"缺陷类型", "position":"位置区域", "pos_info":"位置筛选依赖当前模型已学习到的语义能力。", "size":"大小", "relation":"与主栅线关系", "final":"模型基础查询", "manual":"可选手动真实 GT 掩码", "gt_note":"自动 GT 需要原始数据集图片和精确数据集表达式；手动 GT 始终优先。", "start":"开始检测", "clear":"清空", "gt":"真实 GT", "pred":"预测缺陷", "compare":"GT 与预测对比", "overlay":"预测叠加原图", "metrics":"### 评估指标", "info":"## 推理信息", "model":"模型信息", "download":"下载结果", "locked":"由文本确定 / 已锁定", "detected":"文本已识别", "supplemented":"手动补充", "none":"未指定", "note":"界面语言可切换；模型文本输入建议使用英文，以保持与训练数据一致。"},
}
POS = ["Not specified","left","right","top","bottom","center","upper-left","upper-right","lower-left","lower-right"]
SIZES = ["Not specified","tiny","little","small","large"]
RELS = ["Not specified","not connected to busbar","connected to busbar","not intersected with busbar","intersected with busbar","not touching busbar","touching busbar"]
ZH = {"Not specified":"未指定","left":"左侧","right":"右侧","top":"上部","bottom":"下部","center":"中心","upper-left":"左上","upper-right":"右上","lower-left":"左下","lower-right":"右下","tiny":"极小","little":"较小（little）","small":"较小（small）","large":"大","not connected to busbar":"未连接主栅线","connected to busbar":"连接主栅线","not intersected with busbar":"未与主栅线相交","intersected with busbar":"与主栅线相交","not touching busbar":"未接触主栅线","touching busbar":"接触主栅线","Resistive Crack":"电阻性裂纹","Closed Crack":"闭合裂纹","Isolated Crack":"孤立裂纹","Disconnected Interconnect":"断开互连","Highly Resistive Interconnect":"高电阻互连","Corrosion Contact":"腐蚀接触","Finger Interruption Contact":"栅指中断接触"}
DEFECTS = ["Not specified"] + (ENGINE.get_structured_options()["defect_types"] if ENGINE else [])

def choices(values, lang):
    return [(ZH.get(v, v), v) for v in values] if lang == "中文" else [(v, v) for v in values]

def parsed(text):
    return ENGINE.parse_expression_fields(str(text or "")) if ENGINE else {"defect_type":None,"position":None,"size":None,"busbar_relation":None}

def compose(text, defect, position, size, relation):
    if not ENGINE: return ""
    return ENGINE.compose_final_query(text, parsed(text), {"defect_type":defect,"position":position,"size":size,"busbar_relation":relation})

def canonical_conditions(text, defect, position, size, relation):
    """Return the same locked-or-supplemented values shown by the UI."""
    p=parsed(text)
    return {
        "defect_type":p["defect_type"] or (defect if defect in DEFECTS else "Not specified"),
        "position":p["position"] or (position if position in POS else "Not specified"),
        "size":p["size"] or (size if size in SIZES else "Not specified"),
        "busbar_relation":p["busbar_relation"] or (relation if relation in RELS else "Not specified"),
    }

def query_plan(text, defect, position, size, relation):
    """Keep text semantics in the whole-image model pass; filter later choices."""
    values=canonical_conditions(text,defect,position,size,relation)
    parsed_values=parsed(text)
    if str(text or "").strip():
        base_conditions={key:(parsed_values.get(key) or "Not specified") for key in values}
        supplements={key:value for key,value in values.items() if not parsed_values.get(key) and value!="Not specified"}
    else:
        # With no natural-language source, structured conditions define the
        # only available model query and are therefore not post-filters.
        base_conditions=values; supplements={}
    base_query=ENGINE.compose_final_query(text,parsed_values,base_conditions) if ENGINE else ""
    return base_query, values, supplements

def field(value, locked, values, lang, label):
    value = locked if locked else (value if value in values else "Not specified")
    suffix = f" ({I18N[lang]['locked']})" if locked else ""
    return gr.Dropdown(choices=choices(values, lang), value=value, interactive=not bool(locked), label=label + suffix)

def semantic(p, vals, lang):
    t=I18N[lang]; labels={"defect_type":t["defect"],"position":t["position"],"size":t["size"],"busbar_relation":t["relation"]}
    found=[f"- {labels[k]}: `{v}`" for k,v in p.items() if v]
    added=[f"- {labels[k]}: `{vals[k]}`" for k in labels if not p[k] and vals[k] != "Not specified"]
    return f"**{t['detected']}:**\n" + ("\n".join(found) or "- " + t["none"]) + f"\n\n**{t['supplemented']}:**\n" + ("\n".join(added) or "- " + t["none"]) + "\n\n" + ("**Supplementary Filters:** " if lang=="English" else "**补充筛选条件：** ") + (", ".join(f"{k}={v}" for k,v in vals.items() if not p.get(k) and v!="Not specified") or t["none"])

def sync(text, defect, position, size, relation, lang):
    lang=lang if lang in I18N else "English"; p=parsed(text)
    vals={"defect_type":p["defect_type"] or (defect if defect in DEFECTS else "Not specified"),"position":p["position"] or (position if position in POS else "Not specified"),"size":p["size"] or (size if size in SIZES else "Not specified"),"busbar_relation":p["busbar_relation"] or (relation if relation in RELS else "Not specified")}
    ds=[field(vals["defect_type"],p["defect_type"],DEFECTS,lang,I18N[lang]["defect"]),field(vals["position"],p["position"],POS,lang,I18N[lang]["position"]),field(vals["size"],p["size"],SIZES,lang,I18N[lang]["size"]),field(vals["busbar_relation"],p["busbar_relation"],RELS,lang,I18N[lang]["relation"])]
    base_query, _, _ = query_plan(text, vals["defect_type"], vals["position"], vals["size"], vals["busbar_relation"])
    return (*ds, base_query, semantic(p,vals,lang))

def metrics(info, result, lang):
    t=I18N[lang]; m=result["metrics"]
    if m.get("Pixel Accuracy") is None:
        return f"{t['metrics']}\n**GT:** {info.get('GT Source','Unavailable')}  \n**Status:** {info.get('GT Match Status','N/A')}  \n\nAccuracy / Dice / IoU / Precision / Recall / Specificity / F1: N/A (no verified GT)."
    f=lambda k: "N/A" if m.get(k) is None else f"{float(m[k]):.4f}"
    region=info.get("Evaluation Region","Full image")
    gt_state=info.get("GT in Region","N/A")
    pred_state=info.get("Prediction in Region","N/A")
    result_state=info.get("Region Result","N/A")
    if lang=="中文":
        localized={"Full image":"整张图","Present":"存在目标","Empty":"无目标","Correct No-target":"正确无目标","False Activation":"误检 / 错误激活","Exact Match":"完全匹配","Partial Match":"部分匹配","GT unavailable":"GT 不可用"}
        region=localized.get(region,region); gt_state=localized.get(gt_state,gt_state); pred_state=localized.get(pred_state,pred_state); result_state=localized.get(result_state,result_state)
    note=""
    if result_state=="Correct No-target": note="\n\nNo foreground target exists in this evaluation region."
    elif result_state=="False Activation": note=f"\n\nBackground-dominated accuracy: no GT foreground exists in this region. False Positive Pixels: **{m.get('FP Pixels',0)}**; False Activation Area Ratio: **{float(info.get('False Activation Area Ratio') or 0)*100:.3f}%**."
    labels=("当前评价区域","区域 GT 状态","区域 Prediction 状态","区域结果") if lang=="中文" else ("Evaluation Region","GT in Region","Prediction in Region","Region Result")
    return f"{t['metrics']}\n**{labels[0]}:** {region}  \n**{labels[1]}:** {gt_state}  \n**{labels[2]}:** {pred_state}  \n**{labels[3]}:** {result_state}\n\n| Accuracy | Dice | IoU | Precision | Recall | Specificity | F1 |\n|---:|---:|---:|---:|---:|---:|---:|\n| **{float(m['Pixel Accuracy'])*100:.2f}%** | {f('Dice')} | {f('IoU')} | {f('Precision')} | {f('Recall')} | {f('Specificity')} | {f('F1')} |{note}"

def detect(image,text,defect,position,size,relation,_final,gt,lang):
    if not ENGINE: return (None,None,None,None,"### Evaluation Metrics\nN/A",[],None,f"Model initialization failed: `{ENGINE_ERROR}`")
    if image is None: return (None,None,None,None,"### Evaluation Metrics\nN/A",[],None,"No image uploaded.")
    try:
        expression,conditions,supplements=query_plan(text,defect,position,size,relation)
        # Only this base whole-image query reaches BERT/model inference.
        result=ENGINE.infer(image=image,expression=expression,gt_mask=gt,query_conditions=conditions,original_expression=text); info=dict(result["info"]); info["Generated Expression"]=expression
        info["Supplementary Filters"] = ", ".join(f"{k}={v}" for k,v in supplements.items()) or "None"
        return result["ground_truth"],result["binary_mask"],result["gt_vs_prediction"],result["overlay"],metrics(info,result,lang),ENGINE.info_rows(info),result["download_zip"],f"**{result['status']}**\n\n**Base Model Query:** `{expression}`\n\n**Supplementary Filters:** `{info['Supplementary Filters']}`\n\n**GT:** {info.get('GT Match Status','N/A')}"
    except Exception as exc:
        traceback.print_exc(); return (None,None,None,None,"### Evaluation Metrics\nN/A",[],None,f"Inference failed: `{type(exc).__name__}: {exc}`")

def reset(lang):
    return None,"","Not specified","Not specified","Not specified","Not specified",compose("","Not specified","Not specified","Not specified","Not specified"),None,None,None,None,None,I18N[lang]["metrics"]+"\nNo result yet.",[],None,"",semantic(parsed(""),{"defect_type":"Not specified","position":"Not specified","size":"Not specified","busbar_relation":"Not specified"},lang)

def legend_html(lang):
    if lang=="中文":
        title="GT 与预测对比"; entries=(("green","绿色=正确重合 (TP)"),("red","红色=误检 (FP)"),("blue","蓝色=漏检 (FN)"))
    else:
        title="GT vs Prediction"; entries=(("green","Green = correct overlap (TP)"),("red","Red = false positive (FP)"),("blue","Blue = missed GT defect (FN)"))
    content="&nbsp;&nbsp;".join(f'<span class="legend-swatch {color}"></span>{label}' for color,label in entries)
    return f'<div class="gt-legend"><b>{title}:</b>&nbsp;&nbsp;{content}</div>'

def localize(lang,text,defect,position,size,relation):
    t=I18N[lang]; q=sync(text,defect,position,size,relation,lang)
    return (gr.HTML(f'<div class="hero"><h1>{t["title"]}</h1></div>'),gr.Markdown(t["input"]),gr.Markdown(t["result"]),gr.Textbox(label=t["free"],info=t["free_info"]),gr.Markdown(t["structured"]),q[0],q[1],q[2],q[3],gr.Markdown(t["pos_info"]),gr.Textbox(label=t["final"],value=q[4]),q[5],gr.Image(label=t["manual"]),gr.Markdown(t["gt_note"]),gr.Button(t["start"]),gr.Button(t["clear"]),gr.HTML(legend_html(lang)),gr.Image(label=t["gt"]),gr.Image(label=t["pred"]),gr.Image(label=t["compare"]),gr.Image(label=t["overlay"]),gr.File(label=t["download"]),gr.Markdown(t["info"]),gr.Accordion(label=t["model"]),gr.Markdown(t["note"]))

CSS="""
.gradio-container { max-width: 1600px !important; margin: 0 auto !important; }
.hero { padding: 18px 22px; border-radius: 14px; background: linear-gradient(135deg, rgba(30,41,59,.95), rgba(15,23,42,.95)); color: white; margin-bottom: 14px; }
.hero h1 { margin: 0 0 6px 0; font-size: 28px; }
.hero p { margin: 0; opacity: .88; }
.small-note { font-size: 13px; opacity: .82; }
.gt-legend { border: 1px solid rgba(148,163,184,.25); border-radius: 10px; padding: 9px 12px; margin: 6px 0 10px 0; font-size: 13px; }
.legend-swatch { display: inline-block; width: 10px; height: 10px; margin: 0 4px 0 0; vertical-align: -1px; border: 1px solid rgba(0,0,0,.35); }
.legend-swatch.green { background: #00b050; }.legend-swatch.red { background: #e53935; }.legend-swatch.blue { background: #1e88e5; }
"""
with gr.Blocks(title=I18N["English"]["title"],css=CSS) as demo:
    language=gr.Radio(["English","中文"],value="English",label="Language / 语言")
    hero=gr.HTML(f'<div class="hero"><h1>{I18N["English"]["title"]}</h1><p>Frozen General Model + Presence Gate v2 + Ultra / Tiny / Larger Expert Routing + Automatic Dataset Ground Truth Evaluation</p></div>')
    with gr.Row():
      with gr.Column(scale=5):
        in_head=gr.Markdown(I18N["English"]["input"]); image=gr.Image(label="Upload EL Image",type="filepath",image_mode="RGB")
        text=gr.Textbox(label=I18N["English"]["free"],lines=2,info=I18N["English"]["free_info"]); struct_head=gr.Markdown(I18N["English"]["structured"])
        with gr.Row(): defect=gr.Dropdown(choices=choices(DEFECTS,"English"),value="Not specified",label=I18N["English"]["defect"]); position=gr.Dropdown(choices=choices(POS,"English"),value="Not specified",label=I18N["English"]["position"])
        pos_note=gr.Markdown(I18N["English"]["pos_info"])
        with gr.Row(): size=gr.Dropdown(choices=choices(SIZES,"English"),value="Not specified",label=I18N["English"]["size"]); relation=gr.Dropdown(choices=choices(RELS,"English"),value="Not specified",label=I18N["English"]["relation"])
        final=gr.Textbox(label=I18N["English"]["final"],value=compose("","Not specified","Not specified","Not specified","Not specified"),interactive=False,lines=3); sem=gr.Markdown()
        gt=gr.Image(label=I18N["English"]["manual"],type="filepath"); gt_note=gr.Markdown(I18N["English"]["gt_note"])
        with gr.Row(): start=gr.Button(I18N["English"]["start"],variant="primary"); clear=gr.Button(I18N["English"]["clear"])
      with gr.Column(scale=5):
        out_head=gr.Markdown(I18N["English"]["result"]); status=gr.Markdown()
        comparison_legend=gr.HTML(legend_html("English"))
        with gr.Row(): gt_out=gr.Image(label=I18N["English"]["gt"],interactive=False); pred=gr.Image(label=I18N["English"]["pred"],interactive=False)
        with gr.Row(): compare=gr.Image(label=I18N["English"]["compare"],interactive=False); overlay=gr.Image(label=I18N["English"]["overlay"],interactive=False)
        summary=gr.Markdown(I18N["English"]["metrics"]+"\nNo result yet."); download=gr.File(label=I18N["English"]["download"])
    info_head=gr.Markdown(I18N["English"]["info"]); table=gr.Dataframe(headers=["Metric / Information","Value"],datatype=["str","str"],interactive=False)
    with gr.Accordion(I18N["English"]["model"],open=False) as model: gr.Markdown(ENGINE.model_information_markdown() if ENGINE else ENGINE_ERROR)
    note=gr.Markdown(I18N["English"]["note"])
    inputs=[text,defect,position,size,relation,language]; qouts=[defect,position,size,relation,final,sem]
    text.change(sync,inputs=inputs,outputs=qouts)
    for x in [defect,position,size,relation]: x.change(sync,inputs=inputs,outputs=qouts)
    language.change(localize,inputs=[language,text,defect,position,size,relation],outputs=[hero,in_head,out_head,text,struct_head,defect,position,size,relation,pos_note,final,sem,gt,gt_note,start,clear,comparison_legend,gt_out,pred,compare,overlay,download,info_head,model,note])
    start.click(detect,inputs=[image,text,defect,position,size,relation,final,gt,language],outputs=[gt_out,pred,compare,overlay,summary,table,download,status])
    clear.click(reset,inputs=[language],outputs=[image,text,defect,position,size,relation,final,gt,gt_out,pred,compare,overlay,summary,table,download,status,sem])
if __name__=="__main__": demo.launch(server_name="127.0.0.1",server_port=7860,share=False,inbrowser=True,show_error=True)
