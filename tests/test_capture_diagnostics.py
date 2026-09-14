import sqlite3
from pipeline.sfm import write_capture_diagnostics


def test_planar_matches_are_reported_without_claiming_reconstruction(tmp_path):
    with sqlite3.connect(tmp_path/'database.db') as database:
        database.execute('CREATE TABLE two_view_geometries (config INTEGER, rows INTEGER)')
        database.executemany('INSERT INTO two_view_geometries VALUES (?, ?)',[(6,100),(6,120),(0,0)])
    result=write_capture_diagnostics(tmp_path)
    assert result['verified_pairs']==2
    assert result['configurations']==[{'type':'PLANAR_OR_PANORAMIC','pairs':2,'min_matches':100,'max_matches':120}]
    assert (tmp_path/'capture_diagnostics.json').is_file()
