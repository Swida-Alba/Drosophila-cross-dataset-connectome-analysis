"""Offline PowerPoint export: native DrawingML geometry and label frames.

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
// SVG path commands become DrawingML freeform points, never an image.
// Arcs are split into cubic Beziers (at most 90 degrees per segment).
function pptxPathPoints(data) {
    const tokens=data.match(/[a-zA-Z]|[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?/g)||[];
    let i=0,cmd='',x=0,y=0,startX=0,startY=0,prev='',control=null;const points=[];
    const number=()=>{if(i>=tokens.length||/^[a-z]$/i.test(tokens[i]))throw Error('Incomplete SVG path');return Number(tokens[i++]);};
    const arc=(rx,ry,angle,large,sweep,ex,ey)=>{
        rx=Math.abs(rx);ry=Math.abs(ry);
        if(ex===x&&ey===y)return;
        if(!rx||!ry){points.push({x:ex,y:ey});return;}
        const phi=angle*Math.PI/180,c=Math.cos(phi),s=Math.sin(phi);
        const dx=(x-ex)/2,dy=(y-ey)/2,xp=c*dx+s*dy,yp=-s*dx+c*dy;
        const radius=xp*xp/(rx*rx)+yp*yp/(ry*ry);
        if(radius>1){rx*=Math.sqrt(radius);ry*=Math.sqrt(radius);}
        const factor=(large===sweep?-1:1)*Math.sqrt(Math.max(0,(rx*rx*ry*ry-rx*rx*yp*yp-ry*ry*xp*xp)/(rx*rx*yp*yp+ry*ry*xp*xp)));
        const cxp=factor*rx*yp/ry,cyp=-factor*ry*xp/rx;
        const cx=c*cxp-s*cyp+(x+ex)/2,cy=s*cxp+c*cyp+(y+ey)/2;
        const ux=(xp-cxp)/rx,uy=(yp-cyp)/ry,vx=(-xp-cxp)/rx,vy=(-yp-cyp)/ry;
        let a=Math.atan2(uy,ux),delta=Math.atan2(ux*vy-uy*vx,ux*vx+uy*vy);
        if(!sweep&&delta>0)delta-=2*Math.PI;if(sweep&&delta<0)delta+=2*Math.PI;
        const count=Math.ceil(Math.abs(delta)/(Math.PI/2)),step=delta/count;
        const at=t=>({x:cx+rx*c*Math.cos(t)-ry*s*Math.sin(t),y:cy+rx*s*Math.cos(t)+ry*c*Math.sin(t)});
        const tangent=t=>({x:-rx*c*Math.sin(t)-ry*s*Math.cos(t),y:-rx*s*Math.sin(t)+ry*c*Math.cos(t)});
        for(let n=0;n<count;n++){
            const b=a+step,k=4/3*Math.tan(step/4),p=at(a),q=at(b),u=tangent(a),v=tangent(b);
            points.push({x:n===count-1?ex:q.x,y:n===count-1?ey:q.y,curve:{type:'cubic',x1:p.x+k*u.x,y1:p.y+k*u.y,x2:q.x-k*v.x,y2:q.y-k*v.y}});a=b;
        }
    };
    while(i<tokens.length){
        if(/^[a-z]$/i.test(tokens[i]))cmd=tokens[i++];
        if(!cmd)throw Error('Invalid SVG path');
        const op=cmd.toUpperCase(),rel=cmd!==op,bx=rel?x:0,by=rel?y:0;
        const pair=()=>({x:number()+bx,y:number()+by});let end,c1,c2;
        if(op==='Z'){points.push({close:true});x=startX;y=startY;control=null;prev=op;cmd='';continue;}
        if(op==='M'||op==='L'){end=pair();points.push({...end,...(op==='M'?{moveTo:true}:{})});if(op==='M'){startX=end.x;startY=end.y;cmd=rel?'l':'L';}}
        else if(op==='H'){end={x:number()+bx,y};points.push(end);}
        else if(op==='V'){end={x,y:number()+by};points.push(end);}
        else if(op==='C'||op==='S'){
            c1=op==='C'?pair():((prev==='C'||prev==='S')&&control?{x:2*x-control.x,y:2*y-control.y}:{x,y});
            c2=pair();end=pair();points.push({...end,curve:{type:'cubic',x1:c1.x,y1:c1.y,x2:c2.x,y2:c2.y}});
        }else if(op==='Q'||op==='T'){
            c1=op==='Q'?pair():((prev==='Q'||prev==='T')&&control?{x:2*x-control.x,y:2*y-control.y}:{x,y});
            end=pair();points.push({...end,curve:{type:'quadratic',x1:c1.x,y1:c1.y}});
        }else if(op==='A'){
            const rx=number(),ry=number(),a=number(),large=number(),sweep=number();end=pair();arc(rx,ry,a,large,sweep,end.x,end.y);
        }else throw Error('Unsupported SVG path command '+op);
        x=end.x;y=end.y;control=op==='C'||op==='S'?c2:op==='Q'||op==='T'?c1:null;prev=op;
    }
    return points;
}

// Replace exported triangle polygons by native line-end properties. Preserve
// the rim tip as the actual path endpoint; the head is not a separate shape.
function prepareNativePptxArrows(root) {
    const paths=Array.from(root.querySelectorAll('path'));
    let arrowCount=0;
    for(let i=0;i<paths.length;i++){
        const shaft=paths[i];
        if(!shaft.parentNode||shaft.getAttribute('fill')!=='none'||!shaft.getAttribute('stroke')||shaft.getAttribute('stroke')==='none')continue;
        const points=pptxPathPoints(shaft.getAttribute('d')||'');
        if(points.some(p=>p.close)||points.length<2)continue;
        const start=points[0],end=points[points.length-1];
        for(let j=i+1;j<Math.min(paths.length,i+3);j++){
            const head=paths[j];
            if(!head.parentNode||head.getAttribute('stroke')!=='none'||!head.getAttribute('fill')||head.getAttribute('fill')==='none')break;
            const triangle=pptxPathPoints(head.getAttribute('d')||'');
            const vertices=triangle.filter(p=>!p.close);
            if(vertices.length!==3||!triangle.some(p=>p.close))break;
            let best=null;
            for(let k=0;k<3;k++){
                const tip=vertices[k],a=vertices[(k+1)%3],b=vertices[(k+2)%3],mid={x:(a.x+b.x)/2,y:(a.y+b.y)/2};
                for(const [side,p] of [['start',start],['end',end]]){
                    const dist=Math.hypot(mid.x-p.x,mid.y-p.y);
                    if(!best||dist<best.dist)best={side,dist,tip};
                }
            }
            if(!best||best.dist>Math.max(2,parseFloat(shaft.getAttribute('stroke-width'))||1))break;
            const sm=shaft.getCTM(),hm=head.getCTM();
            if(['a','b','c','d','e','f'].some(k=>Math.abs(sm[k]-hm[k])>1e-6))break;
            shaft.setAttribute('data-vp-arrow-'+best.side,'triangle');
            shaft.setAttribute('data-vp-edge','1');
            const endpoint=best.side==='start'?start:end;endpoint.x=best.tip.x;endpoint.y=best.tip.y;
            head.remove();arrowCount++;
        }
        // Retain the exact quadratic/cubic control points and subpath order.
        shaft.__pptxPoints=points;
    }
    return arrowCount;
}

function addNativePptxGeometry(slide,root,scale) {
    let count=0;const lineStyles=[];
    const color=value=>{
        if(!value||value==='none')return null;
        if(/^#[0-9a-f]{6}$/i.test(value))return value.slice(1);
        if(/^#[0-9a-f]{3}$/i.test(value))return value.slice(1).split('').map(c=>c+c).join('');
        const rgb=value.match(/^rgba?\(\s*([\d.]+)[, ]+([\d.]+)[, ]+([\d.]+)/i);
        if(rgb)return rgb.slice(1).map(n=>Math.round(Number(n)).toString(16).padStart(2,'0')).join('');
        throw Error('Unsupported SVG colour '+value);
    };
    for(const el of root.querySelectorAll('path,rect,circle,ellipse,line,polygon,polyline,image,use,foreignObject')){
        if(el.closest('defs,clipPath,mask,marker'))continue;
        if(['image','use','foreignObject'].includes(el.localName))throw Error('Native PowerPoint export does not support SVG '+el.localName+' objects.');
        const css=getComputedStyle(el);if(css.display==='none'||css.visibility==='hidden')continue;
        let opacity=1;for(let n=el;n&&n!==root.parentElement;n=n.parentElement){const v=parseFloat(getComputedStyle(n).opacity);if(Number.isFinite(v))opacity*=v;}
        if(opacity===0)continue;
        const matrix=el.getCTM();if(!matrix)continue;
        const sx=Math.hypot(matrix.a,matrix.b),sy=Math.hypot(matrix.c,matrix.d);
        const fill=color(css.fill),stroke=color(css.stroke);
        const opts={objectName:(el.getAttribute('data-vp-edge')?'Arrow / edge ':'Shape ')+(++count),
            fill:fill?{color:fill,transparency:100*(1-opacity*(parseFloat(css.fillOpacity)||0))}:{color:'FFFFFF',transparency:100},
            line:stroke?{color:stroke,width:(parseFloat(css.strokeWidth)||1)*Math.sqrt(sx*sy)*scale*72,transparency:100*(1-opacity*(parseFloat(css.strokeOpacity)||0)),dashType:css.strokeDasharray==='none'?'solid':'dash'}:{color:'FFFFFF',transparency:100,width:0.01}};
        if(stroke)lineStyles.push({name:opts.objectName,width:parseFloat(css.strokeWidth)||1,
            cap:css.strokeLinecap,dashes:css.strokeDasharray==='none'?[]:css.strokeDasharray.split(/[ ,]+/).map(parseFloat).filter(Number.isFinite)});
        const attr=(key,fallback=0)=>parseFloat(el.getAttribute(key))||fallback;
        // Native preset shapes retain the familiar PowerPoint resize/shape controls.
        if(['rect','circle','ellipse'].includes(el.localName)&&Math.abs(matrix.b)<1e-9&&Math.abs(matrix.c)<1e-9&&matrix.a>0&&matrix.d>0){
            const isRect=el.localName==='rect',rx=attr('rx',attr('r')),ry=attr('ry',attr('r'));
            const left=isRect?attr('x'):attr('cx')-rx,top=isRect?attr('y'):attr('cy')-ry;
            const p=new DOMPoint(left,top).matrixTransform(matrix);
            const w=(isRect?attr('width'):2*rx)*sx*scale,h=(isRect?attr('height'):2*ry)*sy*scale;
            if(w>0&&h>0)slide.addShape(isRect?(rx?'roundRect':'rect'):'ellipse',{...opts,x:p.x*scale,y:p.y*scale,w,h,...(isRect&&rx?{rectRadius:Math.min(rx*sx*scale,w/2,h/2)}:{})});
            continue;
        }
        if(stroke){opts.line.beginArrowType=el.getAttribute('data-vp-arrow-start')||undefined;opts.line.endArrowType=el.getAttribute('data-vp-arrow-end')||undefined;}
        let points;
        if(el.localName==='path')points=el.__pptxPoints||pptxPathPoints(el.getAttribute('d')||'');
        else if(el.localName==='line')points=[{x:attr('x1'),y:attr('y1'),moveTo:true},{x:attr('x2'),y:attr('y2')}];
        else if(el.localName==='polygon'||el.localName==='polyline'){
            const numbers=(el.getAttribute('points')||'').match(/[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:e[-+]?\d+)?/gi)||[];
            points=[];for(let i=0;i<numbers.length;i+=2)points.push({x:Number(numbers[i]),y:Number(numbers[i+1]),...(i===0?{moveTo:true}:{})});
            if(el.localName==='polygon')points.push({close:true});
        }else{
            // Rotated/skewed ellipses/rectangles are still native freeforms.
            let d;
            if(el.localName==='rect'){
                const x=attr('x'),y=attr('y'),w=attr('width'),h=attr('height'),r=Math.min(attr('rx'),w/2,h/2);
                d=`M${x+r} ${y}H${x+w-r}Q${x+w} ${y} ${x+w} ${y+r}V${y+h-r}Q${x+w} ${y+h} ${x+w-r} ${y+h}H${x+r}Q${x} ${y+h} ${x} ${y+h-r}V${y+r}Q${x} ${y} ${x+r} ${y}Z`;
            }else{const x=attr('cx'),y=attr('cy'),rx=attr('rx',attr('r')),ry=attr('ry',attr('r'));d=`M${x-rx} ${y}A${rx} ${ry} 0 1 0 ${x+rx} ${y}A${rx} ${ry} 0 1 0 ${x-rx} ${y}Z`;}
            points=pptxPathPoints(d);
        }
        const transform=(x,y)=>{const p=new DOMPoint(x,y).matrixTransform(matrix);return {x:p.x*scale,y:p.y*scale};};
        const coords=[];
        points=points.map(p=>{if(p.close)return p;const q={...p,...transform(p.x,p.y)};coords.push(q);
            if(p.curve){q.curve={...p.curve};for(const n of [1,2])if(p.curve['x'+n]!==undefined){const c=transform(p.curve['x'+n],p.curve['y'+n]);q.curve['x'+n]=c.x;q.curve['y'+n]=c.y;coords.push(c);}}return q;});
        if(!coords.length)continue;
        const minX=Math.min(...coords.map(p=>p.x)),minY=Math.min(...coords.map(p=>p.y));
        const w=Math.max(1e-6,...coords.map(p=>p.x-minX)),h=Math.max(1e-6,...coords.map(p=>p.y-minY));
        points.forEach(p=>{if(p.close)return;p.x-=minX;p.y-=minY;if(p.curve)for(const n of [1,2])if(p.curve['x'+n]!==undefined){p.curve['x'+n]-=minX;p.curve['y'+n]-=minY;}});
        const straight=points.length===2&&!points.some(p=>p.curve||p.close);
        slide.addShape(straight?'line':'custGeom',{...opts,x:minX,y:minY,w,h,points,
            ...(straight?{flipH:points[1].x<points[0].x,flipV:points[1].y<points[0].y}:{})});
    }
    return {count,lineStyles};
}

// PptxGenJS exposes shape/text primitives but not groups or custom SVG dash
// lengths. Finish the native DrawingML, without introducing picture objects.
async function finishNativePptx(blob,labelCount,pageW,pageH,lineStyles) {
    const zip=await JSZip.loadAsync(await blob.arrayBuffer());
    const file='ppt/slides/slide1.xml',P='http://schemas.openxmlformats.org/presentationml/2006/main',A='http://schemas.openxmlformats.org/drawingml/2006/main';
    const doc=new DOMParser().parseFromString(await zip.file(file).async('string'),'application/xml');
    const make=(ns,name,attrs={})=>{const el=doc.createElementNS(ns,name);for(const [k,v] of Object.entries(attrs))el.setAttribute(k,String(v));return el;};
    const styles=new Map(lineStyles.map(x=>[x.name,x]));
    for(const sp of doc.getElementsByTagNameNS(P,'sp')){
        const name=sp.getElementsByTagNameNS(P,'cNvPr')[0]?.getAttribute('name'),style=styles.get(name);
        if(!style)continue;
        const ln=sp.getElementsByTagNameNS(A,'ln')[0];if(!ln)continue;
        ln.setAttribute('cap',style.cap==='round'?'rnd':style.cap==='square'?'sq':'flat');
        if(style.dashes.length){
            Array.from(ln.children).filter(x=>x.localName==='prstDash'||x.localName==='custDash').forEach(x=>x.remove());
            const dotted=style.dashes[0]<0.1;
            const dash=dotted?make(A,'a:prstDash',{val:'sysDot'}):make(A,'a:custDash');
            if(dotted)ln.setAttribute('cap','rnd');
            else {
                const pattern=style.dashes.length%2?[...style.dashes,...style.dashes]:style.dashes;
                for(let i=0;i<pattern.length;i+=2)dash.appendChild(make(A,'a:ds',{
                    d:Math.max(0,Math.round(pattern[i]/style.width*100000)),sp:Math.max(1,Math.round(pattern[i+1]/style.width*100000))}));
            }
            // In DrawingML the dash definition precedes the line-end properties.
            const before=Array.from(ln.children).find(x=>['round','bevel','miter','headEnd','tailEnd'].includes(x.localName));
            ln.insertBefore(dash,before||null);
        }
    }
    const tree=doc.getElementsByTagNameNS(P,'spTree')[0];
    const objects=Array.from(tree.children).filter(x=>!['nvGrpSpPr','grpSpPr','extLst'].includes(x.localName));
    let nextId=1+Math.max(...Array.from(doc.getElementsByTagNameNS(P,'cNvPr')).map(x=>Number(x.getAttribute('id'))||0));
    const group=(name,objects)=>{
        const g=make(P,'p:grpSp'),nv=make(P,'p:nvGrpSpPr');
        nv.append(make(P,'p:cNvPr',{id:nextId++,name}),make(P,'p:cNvGrpSpPr'),make(P,'p:nvPr'));
        const props=make(P,'p:grpSpPr'),xf=make(A,'a:xfrm');
        const cx=Math.round(pageW*914400),cy=Math.round(pageH*914400);
        xf.append(make(A,'a:off',{x:0,y:0}),make(A,'a:ext',{cx,cy}),make(A,'a:chOff',{x:0,y:0}),make(A,'a:chExt',{cx,cy}));
        props.append(xf);g.append(nv,props);objects.forEach(x=>g.append(x));return g;
    };
    const split=objects.length-labelCount;
    const graphics=group('Graphics',objects.slice(0,split)),labels=group('Labels',objects.slice(split));
    const ext=Array.from(tree.children).find(x=>x.localName==='extLst');tree.insertBefore(graphics,ext||null);tree.insertBefore(labels,ext||null);
    zip.file(file,new XMLSerializer().serializeToString(doc));
    return zip.generateAsync({type:'blob',mimeType:'application/vnd.openxmlformats-officedocument.presentationml.presentation',compression:'DEFLATE'});
}

async function buildEditableNetworkPptx(svgString) {
    if (typeof PptxGenJS === 'undefined') throw Error('Offline PowerPoint export library is unavailable in this build.');
    const parsed = new DOMParser().parseFromString(sanitizeSvgForOffice(svgString,{nativePptx:true}), 'image/svg+xml');
    if (parsed.querySelector('parsererror')) throw Error('Invalid SVG export');
    const root = document.importNode(parsed.documentElement, true);
    const w = parseFloat(root.getAttribute('width'));
    const h = parseFloat(root.getAttribute('height'));
    if (!(w > 0 && h > 0)) throw Error('Invalid SVG dimensions');
    // A live, offscreen SVG supplies the browser's actual font metrics and
    // transforms; no character-count estimates or screen-zoom coordinates.
    const host = document.createElement('div');
    host.style.cssText = 'position:fixed;left:-100000px;top:0;pointer-events:none';
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
        // Every primitive is a native preset shape or DrawingML freeform.
        // No SVG, PNG fallback or picture object is inserted into the slide.
        const arrowCount=prepareNativePptxArrows(root);
        const geometry=addNativePptxGeometry(slide,root,scale);
        labels.forEach(label=>slide.addText(label.text,label.options));
        const blob=await finishNativePptx(await pptx.write({outputType:'blob'}),labels.length,pageW,pageH,geometry.lineStyles);
        return {blob,labelCount:labels.length,shapeCount:geometry.count,arrowCount};
    } finally { host.remove(); }
}

async function exportEditablePPTX() {
    try {
        let svg=cy.svg({full:true,bg:bgCtrl.getColor()});
        if(typeof prepareEditablePptxSvg==='function')svg=prepareEditablePptxSvg(svg);
        const result=await buildEditableNetworkPptx(svg);
        const name=typeof portableNetworkFilename==='function'
            ? portableNetworkFilename().replace(/[.]html$/i,'.pptx')
            : 'network_editable_'+new Date().toISOString().replace(/[-:]/g,'').replace(/[.].*/, '').replace('T','_')+'.pptx';
        downloadBlob(result.blob,name);
        showToast('PowerPoint exported: all shapes, arrows and labels are native editable PowerPoint objects.','success');
    } catch(err) { console.error('PowerPoint export failed',err);showToast('PowerPoint export failed: '+err.message,'error'); }
}
"""
