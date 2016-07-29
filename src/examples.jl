using Distributions
using Winston
using DirectedInformation

function alpha_func(t::Array{Float64,1},a::Real,b::Real,t0::Real)
	y = zeros(t)
	for i in 1:length(y)
		y[i] = alpha_func(t[i], a,b,t0)
	end
	y
end

function alpha_func(t::Float64,a::Real,b::Real,t0::Real)
    y = (1-exp(-(t-t0)./a)).*exp(-(t-t0)./b)
end

logistic(x,β, κ, x0) = 1./(1 + exp(-κ*(β'*x - x0)))

"""
Generate a system of two processes where the source process drives the target process through a logistic coupling. The variable `windowsize` determines the amount of history included in the coupling, while `instcouple` determines whether to include an instantaneous coupling. `κ` and `x0` determine the steepness and the offset of the logistic coupling function, respectively. Finally, `coupling_strength` is the maximum strength of the coupling between the two processes.
"""
function generate_data(;ntrials=1000,windowsize=2,instcouple=false,κ=4.0, x0=0.5,coupling_strength=0.8)
	binsize = 5.0
	t = collect(-50.0:binsize:250)
	A = alpha_func(t, 200.0, 20.0,-50)
	B = zeros(A)
	B[10:end] = A[1:end-9]
	x = zeros(Int64,length(t),ntrials)
	y = zeros(Int64,length(t),ntrials)
	B = coupling_strength*B./maximum(B)
	if instcouple
		offset = 0
	else
		offset = 1
	end
	for j in 1:ntrials
		x[1:windowsize,j] = rand(Bernoulli(0.1), windowsize)
		y[1:windowsize,j] = rand(Bernoulli(0.1), windowsize)
		for i in windowsize+1:size(x,1)
			x[i,j] = rand(Bernoulli(0.1))
			pp = first(1./(1+exp(-κ*(B[i-windowsize:i-offset]'*x[i-windowsize:i-offset,j]-x0))))
			y[i,j] = rand(Bernoulli(pp))
		end
	end
	x,y,B,t
end

function example1(;windowsize=5,kvs...)
	x,y,β,t = generate_data(;windowsize=windowsize, kvs...)
	DI = zeros(size(x,1)-(windowsize-1), 4)
	μ = zeros(DI)
	σ = zeros(DI)
	ta = Winston.Table(5,1)
	aa = zeros(4)
	ta[1,1] = Winston.plot(t[windowsize:end],β[windowsize:end])
	for (i,α) in enumerate([1.0, 1.5, 2.0, 3.0])
		di = DirectedInformation.directed_information(Entropies.MaEstimator, x, y, windowsize;nruns=100,α=α,verbose=0)
		DI[:,i] = di[:,1]
		μ[:,i] = mean(di[:,2:end],2)
		σ[:,i] = std(di[:,2:end],2)
		_t = t[windowsize:end]
		ta[i+1,1] = Winston.FramedPlot()
		Winston.add(ta[i+1,1], Winston.FillBetween(_t, μ[:,i]-σ[:,i], _t, μ[:,i]+σ[:,i]))
		Winston.plot(ta[i+1,1], _t, DI[:,i])
		vv = di[:,1]-(μ[:,i] + σ[:,i])
		aa[i] = sum(vv[vv.>0])./sum(di[:,1])
	end
	ta, DI, μ, σ,aa, sum(β)
end

function plot_data(DI, μ, σ, t)

end
