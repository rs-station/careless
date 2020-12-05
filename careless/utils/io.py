import reciprocalspaceship as rs
import pandas as pd
import gemmi

def load_isomorphous_mtzs(*filenames):
    """
    Simple utility function that loads a batch of isomorphous Mtzs into single rs.DataSet.
    """
    data = []
    print("Loading Mtz files...")
    a,b,c,alpha,beta,gamma=0.,0.,0.,0.,0.,0.
    spacegroup = None
    for i,inFN in enumerate(filenames):
        ds = rs.read_mtz(inFN)
        if spacegroup is not None:
            if ds.spacegroup != spacegroup:
                raise ValueError(f"Filename: {inFN} has space group {ds.spacegroup}, but {spacegroup} is expected.  Cannot load non-isomorphous MTZs.")
        spacegroup = ds.spacegroup
        ds['file_id'] = i
        a += ds.cell.a/len(filenames)
        b += ds.cell.b/len(filenames)
        c += ds.cell.c/len(filenames)
        alpha += ds.cell.alpha/len(filenames)
        beta  += ds.cell.beta/len(filenames)
        gamma += ds.cell.gamma/len(filenames)
        data.append(ds)
    data = pd.concat(data)
    data.cell = gemmi.UnitCell(a, b, c, alpha, beta, gamma)
    data.spacegroup = spacegroup 
    return data

