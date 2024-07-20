for FILE in tests/*.py ; do
    aider --yes --test $FILE --test-cmd "pytest"
done

for FILE in *.py ; do
    aider --yes --message "ensure that ensure RUFF is followed by looking at pyproject and add descriptive docstrings and examples matching the format to all **important** functions.\
      ensure that the examples specify complex image,text nested structure data if applicable. Add composite example using multiple classes at the\
      bottom in a main method" $FILE 
done