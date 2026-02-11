'''dices-common: ancillary code for parsing Perseus texts
'''

import re
import bisect
from copy import deepcopy
import requests
from lxml import etree
import logging
import pandas as pd
import spacy

#
# global values
#

URN_PATTERN = r"urn:cts:greekLit:(tlg\d+)\.(tlg\d+)\.(perseus-grc\d)"
GITHUB_PATTERN = "https://raw.githubusercontent.com/PerseusDL/canonical-greekLit/master/data/{workgroup}/{work}/{workgroup}.{work}.{edition}.xml"
NS_MAP = {
    "cts": "http://chs.harvard.edu/xmlns/cts",
    "tei": "http://www.tei-c.org/ns/1.0",
    "py": "http://codespeak.net/lxml/objectify/pytype",
}
SPACY_MODEL = "grc_odycy_joint_trf"

#
# global objects
#

# turn logging on
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

# initialize spacy parser
nlp = spacy.load(SPACY_MODEL)

#
# class definitions
#

class PerseusText(object):
    '''represents unit of text; one book of a multi-book epic'''
    
    def __init__(self, urn=None, author=None, title=None, prefix=None, populate=True):
        self.urn = urn
        self.author = author
        self.title = title
        self.prefix = prefix
        self._xml = None
        self._line_array = []
        self._line_index = []
        self._spacy_doc = None
        
        if urn and populate:
            self.populate()
    
    
    def __repr__(self):
        if self.author and self.title:
            s = f"{self.author} {self.title}"
            if self.prefix:
                s = s + " " + self.prefix
                
            return f"<Text: {s}>"
            
        elif self.urn:
            return f"Text: {self.urn}>"
        
        else:
            return super().__repr__()
            
    
    def get_text(self):
        '''Return complete text from line array as one long string'''
        if self._line_array:
            return " ".join(l["text"] for l in self._line_array)
        else:
            return ""
    
    
    def populate(self, force=False):
        '''Get the text:
            - download XML from Perseus
            - parse XML and build line array, line index
        '''
        # self._dl_scaife_cts(force=force)
        # self._build_line_array(force=force)
    
    
    def _dl_github_xml(self, force=False):
        '''download from PerseusDL canonical-greekLit repo
           - populates xml attribute
        '''
        
        # fail if no urn
        if not self.urn:
            log.warning(f"can't download xml: {self} has no urn")
            return
        
        # use cached if possible
        if self._xml is not None and not force:
            log.debug("used cached xml: {self}")

        # try to parse url
        print(type(URN_PATTERN), type(self.urn))
        m = re.match(URN_PATTERN, self.urn)
            
        if m is None:
            log.warning(f"can't parse urn: {self}")
            return
        else:
            workgroup = m.group(1)
            work = m.group(2)
            edition = m.group(3)
        
        # try to download
        url = GITHUB_PATTERN.format(workgroup=workgroup, work=work, edition=edition)
        log.info(f"retrieving {url}")
    
        res = requests.get(url)
        if not res.ok:
            log.warning(f"failed to retrieve {url}: {res.status_code} {res.reason}")
            self.xml = None
            return
            
        self._xml = etree.fromstring(res.content)
    
    
    def _build_line_array(self, force=False):
        '''extract verse lines from xml
           - populates line_array
        '''

        # bail if no xml
        if self._xml is None:
            log.warning(f"can't build line array: {self} has no xml")
        
        # use cached if possible
        elif len(self._line_array) and not force:
            log.debug(f"used cached line array: {self}")

        # otherwise build line_array
        else:
            log.debug(f"building line array: {self}")
        
            # start with empty array
            self._line_array = []
        
            # work from copy of xml
            xml = deepcopy(self._xml)
        
            # remove notes
            for note in xml.findall(".//tei:note", namespaces=NS_MAP):
                note.clear(keep_tail=True)
        
            # remove deleted lines
            for del_ in xml.findall(".//tei:del", namespaces=NS_MAP):
                del_.clear(keep_tail=True)
        
            # iterate over verse lines
            for line in xml.findall(".//tei:l", namespaces=NS_MAP):
                n = line.get("n")
                text = "".join(s for s in line.itertext())
                text = re.sub(r"\s+", " ", text).strip()
                self._line_array.append(dict(
                    n = n,
                    text = text,
                ))
            
            log.debug(f"{len(self._line_array)} verse lines: {self}")
        
            # build a line-by-line index of leftmost char position 
            # in the one long string
            log.debug(f"building line index: {self}")

            # start index at zero
            self._line_index = []
            cumsum = 0
        
            # iterate over line array, add length (plus 1 for spaces between lines)
            for line in self._line_array:
                self._line_index.append(cumsum)
                cumsum += len(line["text"]) + 1
    
    
    def tokenize(self, force=False):
        '''tokenize and parse text with spacy
           - populates token_table
        '''
        
        s = self.get_text()
        
        # bail if no text
        if not len(s):
            log.warning(f"Can't run NLP on string of length zero: {self}")
        
        # use cached spacy doc if possible
        elif self._spacy_doc and not force:
            log.debug(f"using cached spacy data: {self}")
            
        # otherwise run NLP
        else:            
            log.info(f"running nlp on string of {len(s)} chars: {self}")
            self._spacy_doc = nlp(s)
            log.debug(f"{len(self._spacy_doc)} tokens: {self}")
    
    
    def get_lines(self):
        '''Returns a Data Frame with one row per line'''
        
        line_table = []
        
        # bail if no line array
        if not self._line_array:
            log.warning(f"Can't export lines, no line array: {self}")
        else:
            for line in self._line_array:
                line_table.append(dict(
                    author = self.author,
                    work = self.title,
                    pref = self.prefix,
                    line = line["n"],
                    text = line["text"],
                ))
        
        return pd.DataFrame(line_table)
    
    
    def get_tokens(self):
        '''Returns a Data Frame with one row per token'''
        
        # build token table
        token_table = []
        
        # bail if no line array
        if not self._line_array:
            log.warning(f"Can't build token table, no line array: {self}")
        elif not self._line_index:
            log.warning(f"Can't build token table, no line index: {self}")
        elif not self._spacy_doc:
            log.warning(f"Can't build token table, no spacy data: {self}")
        else:        
            for tok in self._spacy_doc:
                i = bisect.bisect_right(self._line_index, tok.idx) - 1
                token_table.append(dict(
                    author = self.author,
                    work = self.title,
                    pref = self.prefix,
                    line = self._line_array[i]["n"],
                    token = tok.text,
                    lemma = tok.lemma_,
                    pos = tok.pos_,
                    verbform = ";".join(tok.morph.get("VerbForm")),                
                    mood = ";".join(tok.morph.get("Mood")),
                    tense = ";".join(tok.morph.get("Tense")),                
                    voice = ";".join(tok.morph.get("Voice")),
                    person = ";".join(tok.morph.get("Person")),
                    number = ";".join(tok.morph.get("Number")),
                    case = ";".join(tok.morph.get("Case")),
                    gender = ";".join(tok.morph.get("Gender")),
                ))
            
        return pd.DataFrame(token_table)