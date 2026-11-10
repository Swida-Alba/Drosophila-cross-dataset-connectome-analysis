"""PowerPoint label-frame regression: real browser export and native OOXML."""
import base64
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).parent))
from heatmap_cdp import Page, REQUIRES_CDP
from test_vispath_network import _build_network_html


@REQUIRES_CDP
def test_multiline_labels_export_as_single_native_frames(tmp_path):
    html = _build_network_html(tmp_path / 'network.html')
    with Page(html) as page:
        assert page.wait("return typeof cy !== 'undefined' && typeof PptxGenJS !== 'undefined';", timeout=15)
        assert page.js(r"""
          cy.getElementById('A').data('label','Line one\nLine two');
          cy.getElementById('B').data('label','B one\nB two\nB three');
          cy.nodes().style('text-max-width',300);
          cy.getElementById('A').style({'font-weight':'bold','font-style':'italic','font-family':'Georgia','shape':'round-rectangle'});
          buildEditableNetworkPptx(cy.svg({full:true,bg:'white'})).then(result=>{
            var reader=new FileReader();reader.onload=()=>{window.__pptxData=reader.result;};reader.readAsDataURL(result.blob);
          }).catch(err=>{window.__pptxError=String(err);});
          return true;
        """) is True
        assert page.wait("return !!(window.__pptxData || window.__pptxError);", timeout=30)
        assert page.js("return window.__pptxError || null;") is None
        data = page.js("return window.__pptxData;")
        output = tmp_path / 'editable.pptx'
        output.write_bytes(base64.b64decode(data.split(',')[1]))
        # The toolbar entry point also works without the newer HTML filename
        # helper (e.g. when this change is cherry-picked onto an older viewer).
        assert page.js(r"""
          window.portableNetworkFilename=undefined;
          downloadBlob=(blob,name)=>{window.__downloadName=name;};
          exportEditablePPTX(); return true;
        """) is True
        assert page.wait("return !!window.__downloadName;", timeout=30)
        assert page.js("return window.__downloadName;").endswith('.pptx')
        assert page.errors() == []
    ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
        'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        slide=ET.fromstring(z.read('ppt/slides/slide1.xml'))
        shapes=slide.findall('.//p:sp',ns)
        frames=[f for f in shapes if f.findall('.//a:t',ns)]
        assert len(shapes)>len(frames)
        assert slide.findall('.//a:custGeom',ns)
        assert slide.findall('.//a:cubicBezTo',ns)  # rounded nodes retain native curves
        assert not slide.findall('.//p:pic',ns)
        assert slide.findall('.//a:tailEnd[@type="triangle"]',ns)  # real line arrows, no triangle shapes
        texts=['\n'.join(''.join(p.itertext()) for p in f.findall('p:txBody/a:p',ns)) for f in frames]
        assert texts.count('Line one\nLine two') == 1, texts
        assert texts.count('B one\nB two\nB three') == 1, texts
        assert len(frames)==7, texts  # one native text frame per visible node
        frame=frames[texts.index('Line one\nLine two')]
        assert frame.find('.//a:rPr',ns).get('b')=='1'
        assert frame.find('.//a:rPr',ns).get('i')=='1'
        assert frame.find('.//a:latin',ns).get('typeface')=='Georgia'
        assert not any(n.startswith('ppt/media/') and not n.endswith('/') for n in z.namelist())


@REQUIRES_CDP
def test_native_arrows_dashes_and_separate_groups(tmp_path):
    html = _build_network_html(tmp_path / 'network.html')
    svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="300" height="240">
      <g transform="translate(10 20) scale(2)">
        <rect x="0" y="0" width="100" height="95" rx="5" fill="#ffffff" stroke="#123456"/>
        <path d="M10 10L80 10" fill="none" stroke="#123456" stroke-width="2" stroke-opacity="0.4" stroke-dasharray="6 4"/>
        <path d="M80 5L90 10L80 15Z" fill="#123456" stroke="none"/>
        <path d="M10 40Q40 10 80 40" fill="none" stroke="#abcdef" stroke-width="2" stroke-dasharray="0.01 4" stroke-linecap="round"/>
        <path d="M76.46 43.54L87.07 47.07L83.54 36.46Z" fill="#abcdef" stroke="none"/>
        <path d="M20 70L80 70" fill="none" stroke="#123456" stroke-width="2"/>
        <path d="M80 65L90 70L80 75Z" fill="#123456" stroke="none"/>
        <path d="M20 65L10 70L20 75Z" fill="#123456" stroke="none"/>
        <text x="50" y="80" fill="#123456" font-size="8" text-anchor="middle"><tspan x="50" y="80">Two</tspan><tspan x="50" y="88">lines</tspan></text>
      </g></svg>'''
    import json
    with Page(html) as page:
        assert page.wait("return typeof PptxGenJS !== 'undefined';", timeout=15)
        page.js('''buildEditableNetworkPptx(''' + json.dumps(svg) + ''').then(result=>{
          window.__counts={arrows:result.arrowCount,shapes:result.shapeCount,labels:result.labelCount};
          const reader=new FileReader();reader.onload=()=>window.__nativePptx=reader.result;reader.readAsDataURL(result.blob);
        }).catch(err=>window.__nativeError=String(err));return true;''')
        assert page.wait("return !!(window.__nativePptx || window.__nativeError);", timeout=20)
        assert page.js('return window.__nativeError||null;') is None
        assert page.js('return window.__counts;') == {'arrows':4,'shapes':4,'labels':1}
        raw=base64.b64decode(page.js('return window.__nativePptx;').split(',')[1])
        assert page.errors() == []
    (tmp_path/'native-arrows.pptx').write_bytes(raw)
    import io
    ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
        'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        slide=ET.fromstring(z.read('ppt/slides/slide1.xml'))
        tree=slide.find('p:cSld/p:spTree',ns)
        groups=tree.findall('p:grpSp',ns)
        assert [g.find('p:nvGrpSpPr/p:cNvPr',ns).get('name') for g in groups]==['Graphics','Labels']
        graphics,labels=groups
        assert len(graphics.findall('p:sp',ns))==4  # rect + THREE arrows, no separate head polygons
        assert not graphics.findall('.//a:t',ns)
        assert len(labels.findall('p:sp',ns))==1
        assert [t.text for t in labels.findall('.//a:t',ns)]==['Two','lines']
        assert not slide.findall('.//p:pic',ns)
        assert not any(n.startswith('ppt/media/') and not n.endswith('/') for n in z.namelist())
        arrows=[sp for sp in graphics.findall('p:sp',ns) if sp.findall('.//a:tailEnd',ns)]
        assert len(arrows)==3
        assert len(slide.findall('.//a:headEnd[@type="triangle"]',ns))==1
        assert len(slide.findall('.//a:tailEnd[@type="triangle"]',ns))==3
        dashed,dotted,bidir=arrows
        assert dashed.find('.//a:ds',ns).attrib=={'d':'300000','sp':'200000'}
        assert dotted.find('.//a:prstDash',ns).get('val')=='sysDot'
        assert dotted.find('.//a:custDash',ns) is None
        assert dotted.find('.//a:ln',ns).get('cap')=='rnd'
        assert dashed.find('.//a:ln/a:solidFill/a:srgbClr/a:alpha',ns).get('val')=='40000'
        assert dotted.find('.//a:quadBezTo',ns) is not None
        assert dashed.find('.//a:prstGeom',ns).get('prst')=='line'
        # True target tip lands at SVG x=190 px after nested transforms.
        xfrm=dashed.find('p:spPr/a:xfrm',ns)
        off=xfrm.find('a:off',ns);ext=xfrm.find('a:ext',ns)
        assert abs(int(off.get('x'))+int(ext.get('cx'))-190/300*10*914400)<=2
        assert len({n.get('id') for n in slide.findall('.//p:cNvPr',ns)})==len(slide.findall('.//p:cNvPr',ns))
