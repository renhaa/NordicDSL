for f in sec/*.tex; do echo $f; aspell check $f; done
