# -*- coding: utf-8 -*-
import os
import glob

from ConfigParser import SafeConfigParser
from argparse import Namespace
from ast import literal_eval

def _parse_value(value):
    # Check for boolean or None
    if value.capitalize().startswith(('False', 'True', 'None')):
        return eval(value.capitalize(), {}, {})

    # Check for path, files
    elif value.startswith(('~', '/', '../', './')):
        real_path = os.path.realpath(os.path.expanduser(value))
        if '*' in real_path:
            # Resolve wildcards if any
            files = glob.glob(real_path)
            if len(files) == 0:
                raise Exception('%s did not match any file.' % value)
            # Return list if multiple, single file if not
            return sorted(files) if len(files) > 1 else files[0]
        else:
            return real_path

    else:
        # Detect strings, floats and ints
        try:
            # If this fails, this is a string
            literal = literal_eval(value)
        except ValueError as ve:
            return value
        else:
            # Did not fail => literal is a float or int now
            return literal

def _get_section_dict(l):
    """l is a list of key-value tuples returned by ConfigParser.items().
    Convert it to a dictionary after inferring value types."""
    return {key : _parse_value(value) for key,value in l}

def _update_dict(d, defs):
    """Update d with key-values from defs IF key misses from d."""
    for k,v in defs.items():
        if k not in d:
            d[k] = v
    return d

class Config(SafeConfigParser, object):
    """Custom parser inheriting from SafeConfigParser."""

    def __init__(self, filename, trdefs=None, mddefs=None, override=None):
        # Call parent's __init__()
        super(self.__class__, self).__init__()

        # Use values from defaults.py when missing
        self._trdefs    = trdefs if trdefs else {}
        self._mddefs    = mddefs if mddefs else {}

        # dict that will override
        # this can contain both model and training args unfortunately.
        self._override  = _get_section_dict(override.items()) \
                                if override else {}

        # Parse the file, raise if error
        parsed = self.read(filename)
        if len(self.read(filename)) == 0:
            raise Exception('Could not parse configuration file.')

    def parse(self):
        """Parse everything and return 2 Namespace objects."""
        # Convert training and model sections to dictionary
        trdict = _get_section_dict(self.items('training')) \
                    if 'training' in self.sections() else {}
        mddict = _get_section_dict(self.items('model')) \
                    if 'model' in self.sections() else {}

        # Update parsed sections with missing defaults
        trdict = _update_dict(trdict, self._trdefs)
        mddict = _update_dict(mddict, self._mddefs)

        for key, value in self._override.items():
            assert not (key in trdict and key in mddict)
            if key in trdict:
                trdict[key] = value
            else:
                # everything else goes to model args
                mddict[key] = value

        # Finally merge model.* subsections into model
        for section in self.sections():
            if section.startswith('model.'):
                subsection = section.split('.')[-1]
                mddict[subsection] = _get_section_dict(self.items(section))

        return (Namespace(**trdict), Namespace(**mddict))
