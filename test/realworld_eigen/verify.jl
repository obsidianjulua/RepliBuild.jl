using Test

push!(LOAD_PATH, joinpath(@__DIR__, "julia"))
using RealworldEigen

@testset "Eigen Integration" begin
    @testset "Linkage" begin
        @test RealworldEigen.test_eigen_linkage() == 42
    end

    @testset "create_random_matrix" begin
        mat = RealworldEigen.create_random_matrix(3, 3)
        @test sizeof(mat) == 24
        @test typeof(mat) == RealworldEigen.Matrix_double_minus_1_minus_1_0_minus_1_minus_1

        # Test property accessor — drill into DenseStorage
        storage = mat.m_storage
        @test typeof(storage) == RealworldEigen.DenseStorage_double_minus_1_minus_1_minus_1_0
        @test storage.m_rows == 3
        @test storage.m_cols == 3
        @test storage.m_data != C_NULL

        # Read first element through the data pointer
        val = unsafe_load(storage.m_data)
        @test isa(val, Float64)
        @test isfinite(val)
    end

    @testset "init_model" begin
        model = RealworldEigen.init_model(4)
        @test sizeof(model) == 48
        @test typeof(model) == RealworldEigen.ModelData

        # Test primitive accessor
        @test model.learning_rate == 0.01
    end

    @testset "add_matrices" begin
        a = RealworldEigen.create_random_matrix(2, 2)
        b = RealworldEigen.create_random_matrix(2, 2)
        c = RealworldEigen.Matrix_double_minus_1_minus_1_0_minus_1_minus_1()
        RealworldEigen.add_matrices(Ref(a), Ref(b), Ref(c))
        @test sizeof(c) == 24
        @test typeof(c) == RealworldEigen.Matrix_double_minus_1_minus_1_0_minus_1_minus_1
    end
end
