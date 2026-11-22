from eval.vqa_mac_headroom import available_bytes

VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                    15708.
Pages active:                                 345152.
Pages inactive:                               341632.
Pages speculative:                              2560.
Pages wired down:                             161929.
"""


def test_available_counts_free_inactive_speculative():
    assert available_bytes(VM_STAT) == 16384 * (15708 + 341632 + 2560)
