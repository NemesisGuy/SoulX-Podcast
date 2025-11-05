import inspect, sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from gradio_app import infer_from_ui
print('isgeneratorfunction:', inspect.isgeneratorfunction(infer_from_ui))
try:
    gen = infer_from_ui(None,'[S1]hi', None, None, False, '', None, '', '', False, '', None, None, 'outputs/x.wav', 'hf', False, 42, None, False, False, stream_enabled=True)
    import types
    print('returned type:', type(gen))
    print('is generator instance:', isinstance(gen, types.GeneratorType))
except Exception as e:
    print('call error:', e)
