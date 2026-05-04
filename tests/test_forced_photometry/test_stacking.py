def test_stacking(database_sessionmaker, mapset_with_sources):
    imaps, sources = mapset_with_sources

    with database_sessionmaker() as session:
        session.add_all(imaps)
        session.commit()

    # stacker = ForcedPhotometryStacker(catalogs=[sources])
