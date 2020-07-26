from collections import OrderedDict
import re

from . import scale

class Axis:
    def __init__(self, title_boxes):
        texts = [t.text for t in title_boxes]
        self.title = '|'.join(texts)

    def gen(self):
        axs = OrderedDict()
        axs['title'] = self.title
        return axs

class Channel:
    def __init__(self, spec_width, spec_heigth, title_boxes, label_boxes, orientation):
        
        self.title_boxes = [t.to_dict() for t in title_boxes]
        self.label_boxes = [t.to_dict() for t in label_boxes]

        self.spec_width = spec_width
        self.spec_heigth = spec_heigth

        self.orientation = orientation 

        self.field = ''
        self.type = self.infer_type()
        self.axis = Axis(title_boxes)
        self.scale = scale.Scale(self, self.label_boxes)

    def infer_type(self):
        # if not labels
        if not self.label_boxes:
            return ''

        # check if quantitative
        if self.is_quantitative():
            return 'quantitative'

        # check if temporal
        if self.is_temporal():
            return 'temporal'

        # check if ordinal

        # by default nominal
        return 'nominal'

    def is_quantitative(self):
        # CASES:
        # [1, 2, 3, 4, 5]      # parse to number
        # [1e10, 2e10, 3e10]   # scientific notation
        # [1k, 1M, 1G, 1T]     # SI standard
        try:
            for box in self.label_boxes:
                box['number'] = self.si_parse(box['text'])

            return True
        except ValueError:
            pass

        # CASES:
        # [1MB, 15MB, 150MB, 1.5GB, 6GB, 15GB]
        try:
            for box in self.label_boxes:
                box['number'] = self.bytes_parse(box['text'])

            return True
        except ValueError:
            pass

        # CASES:
        # [10%, 20%, 30%, 40%]
        # [1GB, 2GB, 3GB, 4GB]
        try:
            suffixes = []
            for box in self.label_boxes:
                prefix, number, suffix = self.parse_label(box['text'])
                box['number'] = number
                box['prefix'] = prefix
                box['suffix'] = suffix
                if suffix == '':
                    suffixes.append(suffix)

            if self.is_unique(suffixes):
                return True
        except ValueError:
            pass

        return False

    def is_temporal(self):
        # CASE
        # "’15","Feb.","Mar.","Apr.","May","Jun.","Jul.","Aug.", "Sep."
        # "2011","2012","2013","2014","2015"
        # "’13", "’14", "’15"
        # a = time.strptime("Nov.", "%b.")
        # time.strftime("%x", a)
        notations = ['%b.', '%b']

        for box in self.label_boxes:
            date_parsed = self.check_date(box['text'])
            if date_parsed is None:
                return False

            box['number'] = date_parsed
            #print date_parsed


        return True

    
    def check_date(self, txt):
        import time
        notations = ['%b.', '%b']
        for nota in notations:
            try:
                #print 'testing', txt
                a = time.strptime(txt, nota)
                return time.strftime("%x", a)
            except ValueError:
                pass

        return None

    def gen(self):
        chn = OrderedDict()
        chn['field'] = self.field
        chn['type'] = self.type
        chn['axis'] = self.axis.gen()
        chn['scale'] = self.scale.gen()
        return chn

    @staticmethod
    def parse_label(label):
        regex = r'[-+]?[0-9]*[\.,]?[0-9]+(?:[eE][-+]?[0-9]+)?'
        results = re.findall(regex, label)

        # assume there is only one number in the label
        if len(results) != 1:
            raise ValueError

        number = results[0]
        start = label.find(number)
        end = start + len(number)
        prefix = label[0:start]
        suffix = label[end:len(label)]

        return prefix, float(number.replace(',', '.')), suffix

    @staticmethod
    def is_unique(lst):
        return not lst or lst.count(lst[0]) == len(lst)

    @staticmethod
    def si_parse(value):
        """
        Based on https://github.com/cfobel/si-prefix/blob/master/si_prefix/__init__.py

        Parse a value expressed using SI prefix units to a floating 4point number.
        Args:
            value (str) : Value expressed using SI prefix units (as returned by
                `si_format` function).
        """
        SI_PREFIX_UNITS = 'yzafpnum kMGTPEZY'
        CRE_10E_NUMBER = re.compile(r'^\s*(?P<integer>[\+\-]?\d+)?'
                                    r'(?P<fraction>.\d+)?\s*([eE]\s*'
                                    r'(?P<expof10>[\+\-]?\d+))?$')
        CRE_SI_NUMBER = re.compile(r'^\s*(?P<number>(?P<integer>[\+\-]?\d+)?'
                                   r'(?P<fraction>.\d+)?)\s*'
                                   r'(?P<si_unit>[%s])?\s*$' % SI_PREFIX_UNITS)
        match = CRE_10E_NUMBER.match(value)
        if match:
            # Can be parse using `float`.
            # assert (match.group('integer') is not None or
            #         match.group('fraction') is not None)
            return float(value)

        match = CRE_SI_NUMBER.match(value)
        if match:
            # assert (match.group('integer') is not None or
            #         match.group('fraction') is not None)
            d = match.groupdict()
            si_unit = d['si_unit'] if d['si_unit'] else ' '
            prefix_levels = (len(SI_PREFIX_UNITS) - 1) // 2
            scale = 10 ** (3 * (SI_PREFIX_UNITS.index(si_unit) - prefix_levels))
            return float(d['number']) * scale

        raise ValueError

    @staticmethod
    def bytes_parse(value):
        """
        Based on https://github.com/cfobel/si-prefix/blob/master/si_prefix/__init__.py
        Check also https://en.wikipedia.org/wiki/Megabyte

        Parse a value expressed using Bytes suffix units to a floating point number.
        Args:
            value (str) : Value expressed using Bytes notation.
        """

        PREFIX_UNITS = 'KMGTPEZY'
        CRE_BYTES_NUMBER = re.compile(r'^\s*(?P<number>(?P<integer>[\+\-]?\d+)?'
                                   r'(?P<fraction>.\d+)?)\s*'
                                   r'(?P<si_unit>[%s]B)?\s*$' % PREFIX_UNITS)
        match = CRE_BYTES_NUMBER.match(value)
        if match:
            # assert (match.group('integer') is not None or
            #         match.group('fraction') is not None)
            d = match.groupdict()
            si_unit = d['si_unit'][:-1] if d['si_unit'] else ' '
            scale = 10 ** (3 * PREFIX_UNITS.index(si_unit))
            return float(d['number']) * scale

        raise ValueError