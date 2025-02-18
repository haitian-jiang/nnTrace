from typing import Optional

def calc_global_batch_id(
    called_times: int,
    dp_size: int,
    dp_rank: int,
    vp_rank: Optional[int] = None,
):
    """
    Calculate the global micro batch id from DP information and PP schedule.
    Can extend this function so that user can provide customized batch id 
    calculation.

    Args:
        called_times: The number of times the forward / backward pass has been 
                      called.  
                      Used for PP schedule.  The first call is 0 rather than 1.
        dp_size: The number of data parallel ranks.
        dp_rank: dp_rank and dp_size are used for canonicalize batch id in DP.
        vp_rank: The virtual pipeline rank.  Not used now, may have future use.
    """
    # interleaved PP schedule
    local_batch_id = called_times

    # DP assignment
    global_batch_id = local_batch_id * dp_size + dp_rank
    return global_batch_id
