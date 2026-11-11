@testset "threading_utils" begin
    # check_threading_status returns (no error)
    CircadianPOMDP.check_threading_status()

    # get_optimal_thread_count
    n = CircadianPOMDP.get_optimal_thread_count()
    @test n isa Int
    @test n >= 1

    # threading_summary
    CircadianPOMDP.threading_summary()
end
