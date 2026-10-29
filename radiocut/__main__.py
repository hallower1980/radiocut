import sys
if len(sys.argv) > 1 and sys.argv[1] not in ("--gui",):
    from radiocut.cli import main
    main()
else:
    from radiocut.gui import main
    main()
