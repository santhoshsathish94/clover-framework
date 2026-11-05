"""List the DLLs a Windows executable or DLL imports (PE import table), recursively within its folder."""
import struct
import sys
from pathlib import Path


def imports(path):
    data = Path(path).read_bytes()
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    nsec = struct.unpack_from("<H", data, pe + 6)[0]
    opt = pe + 24
    magic = struct.unpack_from("<H", data, opt)[0]
    dd = opt + (112 if magic == 0x20B else 96)
    imp_rva = struct.unpack_from("<I", data, dd + 8)[0]
    secs = []
    sh = opt + struct.unpack_from("<H", data, pe + 20)[0]
    for i in range(nsec):
        o = sh + 40 * i
        vsize, va, rsize, rptr = struct.unpack_from("<IIII", data, o + 8)
        secs.append((va, max(vsize, rsize), rptr))

    def off(rva):
        for va, size, rptr in secs:
            if va <= rva < va + size:
                return rva - va + rptr
        return None
    out, o = [], off(imp_rva)
    while o is not None:
        name_rva = struct.unpack_from("<I", data, o + 12)[0]
        if name_rva == 0:
            break
        n = off(name_rva)
        out.append(data[n:data.index(b"\0", n)].decode("ascii", "replace"))
        o += 20
    return out


if __name__ == "__main__":
    root = Path(sys.argv[1])
    seen, todo, system = set(), [root], set()
    folder = root.parent
    while todo:
        p = todo.pop()
        for d in imports(p):
            dl = d.lower()
            if dl in seen:
                continue
            seen.add(dl)
            local = folder / d
            if local.is_file():
                todo.append(local)
            else:
                system.add(d)
    print("not in the folder:", sorted(system, key=str.lower))
