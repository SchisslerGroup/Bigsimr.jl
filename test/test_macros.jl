using Test

"""
    @test_isdefined s

Tests whether variable `s` is defined in the current scope.

## Examples

```julia-repl
julia> @test_isdefined newvar
The symbol 'newvar' is not defined
Test Failed at ...
  Expression: false

ERROR: There was an error during testing

julia> newvar = 1

julia> @test_isdefined newvar
Test Passed

julia> function f end
f (generic function with 0 methods)

julia> @test_isdefined f
Test Passed
```
"""
macro test_isdefined(ex)
    fail_msg = "The symbol '$ex' is not defined"
    return quote
        if @isdefined $ex
            @test true
        else
            println($fail_msg)
            @test false
        end
    end
end


"""
    @test_hasmethod f args kws
"""
macro test_hasmethod(f, args, kws::Tuple{Vararg{Symbol}})
    fail_msg = "The method '$f' does not support arguments $args and keywords $kws"
    return quote
        if hasmethod($f, $args, $kws)
            @test true
        else
            println($fail_msg)
            @test false
        end
    end
end

macro test_hasmethod(f, args)
    fail_msg = "The method '$f' does not support arguments $args"
    return quote
        if hasmethod($f, $args)
            @test true
        else
            println($fail_msg)
            @test false
        end
    end
end


"""
    @test_nothrow expr

Test if the expression evaluates successfully, or throws an exception.
"""
macro test_nothrow(ex)
    return quote
        try
            $(esc(ex))
        catch e
            print(e)
            @test false
        else
            @test true
        end
    end
end
