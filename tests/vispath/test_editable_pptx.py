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
          cy.getElementById('A').style({'font-weight':'bold','font-style':'italic','font-family':'Georgia'});
          buildEditableNetworkPptx(sanitizeSvgForOffice(cy.svg({full:true,bg:'white'}))).then(result=>{
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
        frames=slide.findall('.//p:sp',ns)
        texts=['\n'.join(''.join(p.itertext()) for p in f.findall('p:txBody/a:p',ns)) for f in frames]
        assert texts.count('Line one\nLine two') == 1, texts
        assert texts.count('B one\nB two\nB three') == 1, texts
        assert len(frames)==7, texts  # one native text frame per visible node
        frame=frames[texts.index('Line one\nLine two')]
        assert frame.find('.//a:rPr',ns).get('b')=='1'
        assert frame.find('.//a:rPr',ns).get('i')=='1'
        assert frame.find('.//a:latin',ns).get('typeface')=='Georgia'
        svgs=[n for n in z.namelist() if n.endswith('.svg')]
        assert len(svgs)==1
        assert not ET.fromstring(z.read(svgs[0])).findall('.//{http://www.w3.org/2000/svg}text')
        assert any(n.endswith('.png') for n in z.namelist())
