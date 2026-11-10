"""Offline PowerPoint export: convertible SVG geometry + native label frames.

PowerPoint's Convert to Shape splits positioned SVG tspans even when they
belong to one text element. Native DrawingML text is the supported way to
retain an editable multiline label as one text frame.
"""
from pathlib import Path


def pptx_library_script():
    bundle = Path(__file__).resolve().parents[3] / 'src/assets/pptxgenjs/pptxgen.bundle.js'
    if not bundle.exists():
        # Standalone vispath can still use all existing SVG/PNG exports.
        return ''
    return '<script>' + bundle.read_text().replace('</script', '<\\/script') + '</script>'


PPTX_EXPORT_JS = r"""
async function buildEditableNetworkPptx(svgString) {
    if (typeof PptxGenJS === 'undefined') throw Error('Offline PowerPoint export library is unavailable in this build.');
    const parsed = new DOMParser().parseFromString(svgString, 'image/svg+xml');
    if (parsed.querySelector('parsererror')) throw Error('Invalid SVG export');
    const root = document.importNode(parsed.documentElement, true);
    const w = parseFloat(root.getAttribute('width'));
    const h = parseFloat(root.getAttribute('height'));
    if (!(w > 0 && h > 0)) throw Error('Invalid SVG dimensions');
    // A live, offscreen SVG supplies the browser's actual font metrics and
    // transforms; no character-count estimates or screen-zoom coordinates.
    const host = document.createElement('div');
    host.style.cssText = 'position:fixed;left:-100000px;top:0;visibility:hidden;pointer-events:none';
    host.appendChild(root); document.body.appendChild(host);
    const pptx = new PptxGenJS();
    const pageW = Math.min(10, 20 * w / h), pageH = pageW * h / w, scale = pageW / w;
    pptx.defineLayout({name:'NETWORK',width:pageW,height:pageH}); pptx.layout='NETWORK';
    pptx.author='DROCAT'; pptx.subject='Network with editable multiline labels';
    const slide = pptx.addSlide(); const labels=[];
    try {
        if (document.fonts) await document.fonts.ready;
        root.querySelectorAll('text').forEach((text, index) => {
            const box=text.getBBox(), matrix=text.getCTM();
            if (!matrix || !text.textContent) return;
            const spans=Array.from(text.children).filter(x=>x.localName==='tspan');
            const lines=spans.length ? spans.map(x=>x.textContent) : [text.textContent];
            const center=new DOMPoint(box.x+box.width/2,box.y+box.height/2).matrixTransform(matrix);
            const sx=Math.hypot(matrix.a,matrix.b), sy=Math.hypot(matrix.c,matrix.d);
            const fs=parseFloat(text.getAttribute('font-size')) || 12;
            const firstY=spans.length>1 ? parseFloat(spans[0].getAttribute('y')) : NaN;
            const nextY=spans.length>1 ? parseFloat(spans[1].getAttribute('y')) : NaN;
            const linePx=Number.isFinite(nextY-firstY) && nextY>firstY ? nextY-firstY : fs;
            const tw=(box.width+2)*sx*scale;
            const th=Math.max(box.height, linePx*(lines.length-1)+fs*1.25)*sy*scale;
            const anchor=text.getAttribute('text-anchor');
            labels.push({text:lines.join('\n'), options:{
                x:center.x*scale-tw/2,y:center.y*scale-th/2,w:tw,h:th,
                margin:0,vertAnchor:'ctr',align:anchor==='middle'?'center':anchor==='end'?'right':'left',
                fontFace:(text.getAttribute('font-family')||'Arial').split(',')[0].trim().replace(/^['"]|['"]$/g,''),
                fontSize:fs*sy*scale*72,
                bold:text.getAttribute('font-weight')==='bold'||Number(text.getAttribute('font-weight'))>=600,
                italic:['italic','oblique'].includes(text.getAttribute('font-style')),
                color:(text.getAttribute('fill')||'#000000').replace('#',''),
                lineSpacing:linePx*sy*scale*72,paraSpaceAfter:0,paraSpaceBefore:0,
                rotate:Math.atan2(matrix.b,matrix.a)*180/Math.PI,
                objectName:'Label '+(index+1)+': '+lines.join(' / ')
            }});
        });
        // Labels are ONLY native text, never also painted into the SVG.
        root.querySelectorAll('text').forEach(x=>x.remove());
        const geometry = new XMLSerializer().serializeToString(root);
        const bytes=new TextEncoder().encode(geometry);
        let binary=''; for(let i=0;i<bytes.length;i+=8192)binary+=String.fromCharCode(...bytes.subarray(i,i+8192));
        slide.addImage({data:'data:image/svg+xml;base64,'+btoa(binary),x:0,y:0,w:pageW,h:pageH,objectName:'Network geometry — Convert to Shape to edit'});
        labels.forEach(label=>slide.addText(label.text,label.options));
        const blob=await pptx.write({outputType:'blob'});
        return {blob,labelCount:labels.length};
    } finally { host.remove(); }
}

async function exportEditablePPTX() {
    try {
        let svg=cy.svg({full:true,bg:bgCtrl.getColor()});
        if(typeof prepareEditablePptxSvg==='function')svg=prepareEditablePptxSvg(svg);
        const result=await buildEditableNetworkPptx(sanitizeSvgForOffice(svg));
        const name=typeof portableNetworkFilename==='function'
            ? portableNetworkFilename().replace(/[.]html$/i,'.pptx')
            : 'network_editable_'+new Date().toISOString().replace(/[-:]/g,'').replace(/[.].*/, '').replace('T','_')+'.pptx';
        downloadBlob(result.blob,name);
        showToast('PowerPoint exported: one editable text box per label. Convert the geometry image to Shape to edit nodes and edges.','success');
    } catch(err) { console.error('PowerPoint export failed',err);showToast('PowerPoint export failed: '+err.message,'error'); }
}
"""
