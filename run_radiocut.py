import multiprocessing, sys
if __name__ == "__main__":
    multiprocessing.freeze_support()
    if len(sys.argv) > 1:
        from radiocut.cli import main
    else:
        from radiocut.gui import main
    main()
