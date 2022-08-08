import sys
#from absl import flags

def app_run(main=None, argv=None, FLAGS=None):
  """Runs the program with an optional 'main' function and 'argv' list."""
  #f = flags.FLAGS
  f = FLAGS
  
  # Extract the args from the optional `argv` list.
  args = argv[1:] if argv else None
  
  # Parse the known flags from that list, or from the command
  # line otherwise.
  # pylint: disable=protected-access
  flags_passthrough = f._parse_flags(args=args)
  # pylint: enable=protected-access
  
  main = main or sys.modules['__main__'].main
  
  # Call the main function, passing through any arguments
  # to the final program.
  sys.exit(main(sys.argv[:1] + flags_passthrough))
