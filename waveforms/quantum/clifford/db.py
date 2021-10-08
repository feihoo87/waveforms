import sqlite3
from pathlib import Path


def _elms(seq):
    base = {'I'}
    for g1, g2 in zip(*seq):
        if (g1, g2) in [('CZ', 'CZ'), ('iSWAP', 'iSWAP'),
                        ('SQiSWAP', 'SQiSWAP')]:
            base.add(g1)
        elif (g1, g2) in [('C', 'Z'), ('Z', 'C')]:
            base.add('CZ')
        elif g1 == 'C' or g2 == 'C':
            base.add(g1 + g2)
        else:
            base.add(g1)
            base.add(g2)
    return base


class CliffordSequencyDB():
    def __init__(self, db=':memory:'):
        if db != ':memory:':
            self.db = Path(db)
        else:
            self.db = db
        self.conn = sqlite3.connect(':memory:')
        self.i2g = []
        self.g2i = {}
        if self.db == ':memory:' or not self.db.exists():
            self.createDatabase()
        else:
            source = sqlite3.connect(self.db)
            source.backup(self.conn)
        self.loadGateIndex()

    def flush(self):
        if self.db != ':memory:':
            self.conn.backup(sqlite3.connect(self.db))

    def close(self):
        self.flush()
        self.conn.close()

    def __del__(self):
        self.close()

    def createDatabase(self):
        cur = self.conn.cursor()
        # Create table
        cur.execute('''CREATE TABLE gates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name VARCHAR UNIQUE ON CONFLICT IGNORE
        )''')

        cur.execute('''CREATE TABLE sequences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            depth INTEGER,
            clifford_index INTEGER CHECK(clifford_index >= 0 AND clifford_index < 11520),
            seq BLOB(128) UNIQUE ON CONFLICT IGNORE
        )''')

        cur.execute("CREATE INDEX clifford_index ON sequences(clifford_index)")
        cur.execute("CREATE INDEX seq_index ON sequences(seq)")

        cur.execute("INSERT INTO gates VALUES (0, 'I')")
        cur.executemany("INSERT INTO sequences VALUES (?, ?, ?, ?)",
                        [(0, 0, 0, b'')])
        gate_list = [
            'X',
            'Y',
            'Z',
            'X/2',
            '-X/2',
            'Y/2',
            '-Y/2',
            'S',
            '-S',
            'H',
            'CX',
            'XC',
            'CZ',
            'iSWAP',
            'T',
            '-T',
            'SQiSWAP',
        ]

        cur.executemany("INSERT INTO gates (name) VALUES (?)",
                        [(g, ) for g in gate_list])
        self.conn.commit()

    def loadGateIndex(self):
        cur = self.conn.cursor()
        self.i2g = [
            g for g, in cur.execute("SELECT name FROM gates ORDER BY id")
        ]
        self.g2i = {g: i for i, g in enumerate(self.i2g)}

    def updateGateIndex(self, gates):
        gates = set(gates) - set(self.i2g)
        if len(gates) > 0:
            cur = self.conn.cursor()
            cur.executemany("INSERT INTO gates (name) VALUES (?)",
                            [(g, ) for g in gates])
            self.conn.commit()
            self.loadGateIndex()

    def addSeq(self, seq, i):
        self.updateGateIndex(_elms(seq))
        buf, depth = self._seq2bytes(seq)
        self._insertSeq(depth, i, buf)

    def _insertSeq(self, depth, i, buf):
        cur = self.conn.cursor()
        cur.executemany(
            """INSERT INTO sequences
            (depth, clifford_index, seq) VALUES (?, ?, ?)""",
            [(depth, i, buf)])
        self.conn.commit()

    @staticmethod
    def _seqDepth(seq):
        A, B = seq
        if len(A) == 1 and A[0] == B[0] == 'I':
            return 0
        else:
            return len(A)

    def _seq2bytes(self, seq):
        buf = bytearray(128)
        depth = self._seqDepth(seq)
        if depth == 0:
            return buf, 0
        for i, (A, B) in enumerate(zip(*seq)):
            if A == 'C' or B == 'C':
                t = A + B
                A = B = t
            buf[2 * i] = self.g2i[A]
            buf[2 * i + 1] = self.g2i[B]
        return buf, depth

    def _bytes2seq(self, buf, depth):
        if depth == 0:
            return (('I', ), ('I', ))
        A, B = [], []
        for i in range(depth):
            A.append(self.i2g[buf[2 * i]])
            B.append(self.i2g[buf[2 * i + 1]])
        return tuple(A), tuple(B)

    def index2seq(self, i, lim=None):
        cur = self.conn.cursor()
        ret = []
        for buf, depth in cur.execute(f"""SELECT seq, depth FROM sequences
            INDEXED BY clifford_index WHERE clifford_index = {i}
            ORDER BY depth"""):
            ret.append(self._bytes2seq(buf, depth))
        return ret

    def seq2index(self, seq):
        try:
            buf, depth = self._seq2bytes(seq)
            cur = self.conn.cursor()
            q = cur.execute(
                f"""SELECT clifford_index FROM sequences
                    INDEXED BY seq_index
                    WHERE seq=:buf AND depth=:depth""", {
                    'buf': buf,
                    'depth': depth
                })
            return q.fetchone()[0]
        except:
            return None
