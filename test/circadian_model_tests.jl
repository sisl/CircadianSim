@testset "circadian_model" begin
    # Abstract types (API hierarchy)
    @test CircadianPOMDP.CircadianState <: CircadianPOMDP.AbstractCircadianState
    @test CircadianPOMDP.CircadianAction <: CircadianPOMDP.AbstractCircadianAction
    @test CircadianPOMDP.CircadianActionExt <: CircadianPOMDP.AbstractCircadianAction
    @test CircadianPOMDP.CircadianObservation <: CircadianPOMDP.AbstractCircadianObservation

    # model_parameters returns correct structure
    params = CircadianPOMDP.model_parameters()
    @test haskey(params, :params_C)
    @test haskey(params, :params_H)
    @test haskey(params, :params_X)
    @test params.params_C[:tau_x] ≈ 24.2
    @test params.params_C[:K] ≈ 0.55
    @test params.params_H[:chi] ≈ 45.0
    @test params.params_H[:mu_h] ≈ 4.2
    @test params.params_X[:phi] ≈ 0.0

    # get_space_types returns concrete types tuple
    types = CircadianPOMDP.get_space_types()
    @test types == (CircadianPOMDP.CircadianState, CircadianPOMDP.CircadianAction, CircadianPOMDP.CircadianObservation)

    # default_circadian_state
    s = CircadianPOMDP.default_circadian_state()
    @test s isa CircadianPOMDP.CircadianState
    @test s.x ≈ -1.22
    @test s.xc ≈ -0.17
    @test s.H ≈ 11.0
    @test s.tau_x ≈ 24.2
    # Default state may not be on unit circle (depends on model design)

    # default_circadian_action
    a = CircadianPOMDP.default_circadian_action(0.0, 1/60)
    @test a isa CircadianPOMDP.CircadianAction
    @test a.I_act ≈ 0.0
    @test a.dt ≈ 1/60

    a2 = CircadianPOMDP.default_circadian_action(100.0)
    @test a2.I_act ≈ 100.0

    # default_circadian_observation
    o = CircadianPOMDP.default_circadian_observation(0.0)
    @test o isa CircadianPOMDP.CircadianObservation
    @test o.I_obs[1] ≈ 0.0
    @test isempty(o.sleep_onset) && isempty(o.DLMO)
    @test o.startdate < o.stopdate

    # get_all_parameters_flat
    flat = CircadianPOMDP.get_all_parameters_flat()
    @test flat isa Dict
    @test :tau_x in keys(flat)
    @test :chi in keys(flat)
    @test length(flat) > 10

    # print_parameter_summary (no error)
    @test nothing === CircadianPOMDP.print_parameter_summary()
end
