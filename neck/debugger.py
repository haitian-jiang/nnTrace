import pudb.forked
import inspect

def smart_trace(condition=True, trigger_after=1):
    if not condition:
        return
    if not hasattr(smart_trace, '_hit_count'):
        smart_trace._hit_count = 0
    smart_trace._hit_count += 1
    if smart_trace._hit_count < trigger_after:
        return

    frame = inspect.currentframe().f_back
    call_site = (frame.f_code.co_filename, frame.f_lineno)

    if not hasattr(smart_trace, '_hit_cache'):
        smart_trace._hit_cache = set()

    if call_site in smart_trace._hit_cache:
        return
    else:
        smart_trace._hit_cache.add(call_site)
        pudb.forked.set_trace()


__all__ = [
    "smart_trace",
]
