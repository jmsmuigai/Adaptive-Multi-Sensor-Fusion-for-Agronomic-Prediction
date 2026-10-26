"""Produce a tracked-changes .docx (original -> revised) with LibreOffice's
document comparison, run headless over UNO."""
import os
import subprocess
import sys
import time

import uno
from com.sun.star.beans import PropertyValue

orig, rev, out = [os.path.abspath(p) for p in sys.argv[1:4]]


def pv(n, v):
    p = PropertyValue(); p.Name = n; p.Value = v
    return p


proc = subprocess.Popen(['soffice', '--headless', '--invisible', '--norestore',
                         '--accept=socket,host=localhost,port=2002;urp;'])
ctx = None
for _ in range(60):
    try:
        local = uno.getComponentContext()
        res = local.ServiceManager.createInstanceWithContext('com.sun.star.bridge.UnoUrlResolver', local)
        ctx = res.resolve('uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext')
        break
    except Exception:
        time.sleep(1)
smgr = ctx.ServiceManager
desktop = smgr.createInstanceWithContext('com.sun.star.frame.Desktop', ctx)
doc = desktop.loadComponentFromURL(uno.systemPathToFileUrl(rev), '_blank', 0, (pv('Hidden', True),))
disp = smgr.createInstanceWithContext('com.sun.star.frame.DispatchHelper', ctx)
frame = doc.getCurrentController().getFrame()
disp.executeDispatch(frame, '.uno:CompareDocuments', '', 0,
                     (pv('URL', uno.systemPathToFileUrl(orig)), pv('NoAcceptDialog', True)))
time.sleep(3)
doc.storeToURL(uno.systemPathToFileUrl(out), (pv('FilterName', 'MS Word 2007 XML'),))
doc.close(True)
try:
    desktop.terminate()
except Exception:
    pass
proc.wait(timeout=30)
print('saved', out)
