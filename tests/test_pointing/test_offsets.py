from astropy import units as u


def test_apply_offsets(map_with_sources):
    input_map, sources = map_with_sources

    ra_offset = 1.0 * u.arcmin
    dec_offset = -2.0 * u.arcmin
    new_sources = [x.model_copy(deep=True) for x in sources]
    for i in range(len(new_sources)):
        new_sources[i].ra += ra_offset
        new_sources[i].dec += dec_offset

    assert new_sources[0].ra != sources[0].ra
    assert new_sources[0].dec != sources[0].dec
