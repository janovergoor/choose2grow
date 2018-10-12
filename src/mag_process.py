import json
import csv
import os
from util import data_path

"""

  Script to process data for analysis in section 5.3 of the paper.
  Make sure the data has been downloaded first.

  input : ~/mag_raw
  output: data_path/mag-*.csv

"""

path_in = '~/mag_raw'

# dictionary of topics (and keywords)
topics = {
    'imm': {'topic': 'Immunology', 'file': None, 'writer': None},
    'opt': {'topic': 'Optics', 'file': None, 'writer': None},
    'net': {'topic': 'Computer network', 'file': None, 'writer': None},
    'mat': {'topic': 'Materials Science', 'file': None, 'writer': None},
    'com': {'topic': 'Complex systems', 'file': None, 'writer': None},
    'psy': {'topic': 'Psychiatry', 'file': None, 'writer': None},
    'hor': {'topic': 'Horticulture', 'file': None, 'writer': None},
    'cli': {'topic': 'Climatology', 'file': None, 'writer': None}
}

# create file handles
for k, v in topics.items():
    f_out = open('%s/mag_%s.txt' % (data_path, k), 'w')
    writer = csv.writer(f_out, dialect='unix', quoting=csv.QUOTE_NONNUMERIC)
    topics[k]['file'] = f_out
    topics[k]['writer'] = writer

# write headers
header = ["id", "authors", "title", "year", "n_citation",
          "references", "keywords", "lang"]
for v in topics.values():
    tmp = v['writer'].writerow(header)


# Convert author string to initial-last last name,
# so 'Joy M. van Plan' is mapped to 'j-plan'
def clean(s):
    s = s.lower().strip()
    return "%s-%s" % (s[0], s.split(' ')[-1])


# iterate over input files
files = os.listdir(path_in)
files.sort()
for fn in files:
    print(fn)
    f = open("%s/%s" % (path_in, fn), 'r')
    # iterate over lines
    while True:
        line = f.readline()
        if not line:
            break
        # read json
        d = json.loads(line[:-1])
        # skip records without field of study
        if 'fos' not in d:
            continue
        # grab correct writer (if multiple, in order of definition..)
        writer = None
        for k, v in topics.items():
            if v['topic'] in d['fos']:
                writer = v['writer']
        if writer is None:
            continue
        # extract fields
        row = [
            d['id'] if 'id' in d else '',
            ','.join(clean(x['name']) for x in d['authors']) if 'authors' in d else '',
            d['title'] if 'title' in d else '',
            d['year'] if 'year' in d else '',
            d['n_citation'] if 'n_citation' in d else '',
            ','.join(d['references']) if 'references' in d else '',
            ','.join(d['keywords']) if 'keywords' in d else '',
            d['lang'] if 'lang' in d else ''
        ]
        tmp = writer.writerow(row)

# close files
for v in topics.values():
    v['file'].close()
