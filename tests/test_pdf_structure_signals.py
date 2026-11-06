"""PDF structure signals: later revisions (with or without a signature), Info vs XMP disagreement, missing creation date, one typeface under
several subset tags. All documents are fictional and built here."""
from types import SimpleNamespace as NS
import pymupdf
from docforensics.pdf_layer import read_meta, to_utc, parse_xmp
from docforensics.pdf_forensics import analyze_pdf_file, metadata_disagreements, subset_prefix_mismatch

XMP = ('<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
       '<rdf:Description xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmlns:pdf="http://ns.adobe.com/pdf/1.3/">'
       '<xmp:CreatorTool>{creator}</xmp:CreatorTool><pdf:Producer>{producer}</pdf:Producer>'
       '<xmp:CreateDate>{created}</xmp:CreateDate></rdf:Description></rdf:RDF></x:xmpmeta>')


def _pdf(path, incremental=False, xmp=None, info=None):
    d = pymupdf.open()
    d.new_page().insert_text((50, 80), "Fictional test document", fontname="helv", fontsize=12)
    d.set_metadata(info or {"producer": "TestGen 1.0", "creator": "TestGen", "creationDate": "D:20261101100000Z", "modDate": "D:20261101100000Z"})
    if xmp:
        d.set_xml_metadata(xmp)
    d.save(path)
    d.close()
    if incremental:
        d = pymupdf.open(path)
        d[0].insert_text((50, 120), "Added later", fontname="helv", fontsize=12)
        d.saveIncr()
        d.close()


def _docs(path):
    return [NS(doc_id="doc1", path=path, pdf={"spans": []})]


def test_clean_file_has_no_file_history_finding(tmp_path):
    p = str(tmp_path / "clean.pdf")
    _pdf(p)
    assert analyze_pdf_file(_docs(p), read_meta(p), 1) == []


def test_unsigned_incremental_update_is_noted_and_stays_low(tmp_path):
    p = str(tmp_path / "edited.pdf")
    _pdf(p, incremental=True)
    meta = read_meta(p)
    assert meta["revisions"] == 2 and not meta["signed"]
    f = analyze_pdf_file(_docs(p), meta, 1)
    assert len(f) == 1 and f[0].severity == "low" and f[0].confidence < 0.6 and f[0].status == "observed"
    assert "no digital signature" in f[0].explanation and "proves nothing" in f[0].explanation


def test_signed_file_revisions_are_marked_routine():
    meta = {"revisions": 2, "signed": True, "producer": "TestGen", "created": "D:20261101100000Z", "modified": "D:20261101100000Z"}
    f = analyze_pdf_file(_docs("x.pdf"), meta, 1)
    assert len(f) == 1 and "may be routine" in f[0].explanation and "no digital signature" not in f[0].explanation


def test_info_and_xmp_disagreement(tmp_path):
    p = str(tmp_path / "mismatch.pdf")
    _pdf(p, xmp=XMP.format(creator="Microsoft Word", producer="Adobe PDF Library", created="2026-02-01T09:00:00Z"))
    meta = read_meta(p)
    assert meta["xmp"].get("producer") == "Adobe PDF Library"
    notes = metadata_disagreements(meta)
    assert any("producer differs" in n for n in notes)
    assert any("creation date differs" in n for n in notes)


def test_matching_info_and_xmp_is_quiet(tmp_path):
    p = str(tmp_path / "match.pdf")
    _pdf(p, xmp=XMP.format(creator="TestGen", producer="TestGen 1.0", created="2026-11-01T10:00:00Z"))
    assert metadata_disagreements(read_meta(p)) == []


def test_whole_hour_difference_without_timezone_is_ignored():
    meta = {"producer": "A", "created": "D:20261101100000", "xmp": {"created": "2026-11-01T15:30:00"}}
    assert metadata_disagreements(meta) == []                       # 5 h 30 min looks like a time-zone offset, not a disagreement


def test_missing_creation_date_is_noted():
    notes = metadata_disagreements({"producer": "TestGen", "created": None})
    assert any("creation date is missing" in n for n in notes)


def test_font_subset_tags_across_pages():
    spans = lambda tag: {"spans": [{"font": f"{tag}+Arial"}, {"font": "ABCDEF+Roboto"}]}
    docs = [NS(pdf=spans("AAAAAA")), NS(pdf=spans("BBBBBB"))]
    sub = subset_prefix_mismatch(docs)
    assert sub == {"Arial": ["AAAAAA", "BBBBBB"], "Roboto": ["ABCDEF"]} or sub == {"Arial": ["AAAAAA", "BBBBBB"]}
    assert "Roboto" not in sub
    d = NS(doc_id="d", path="m.pdf", pdf=docs[0].pdf)
    f = analyze_pdf_file([d, NS(doc_id="e", path="m.pdf", pdf=docs[1].pdf)], {"revisions": 1}, 1)
    assert len(f) == 1 and "different subset tags" in f[0].explanation and "merged" in f[0].explanation


def test_date_helpers():
    assert to_utc("D:20261103085713+05'30'")[0] == to_utc("2026-11-03T03:27:13Z")[0]
    assert parse_xmp("<a><pdf:Producer>X</pdf:Producer></a>") == {"producer": "X"}
