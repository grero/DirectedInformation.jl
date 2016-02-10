module DirectedInformation
import Entropies
using ArrayViews

"""
Compute the directed information between the variables in `N`. Assumes 

  size(N) == (nbins, ntrials, ncells)

and that `N[:,:,1]` represents the data for the source cell and `N[:,:,2]` prepresents the data for the target cell. The parameter `windowsize` determines how many bins to consider when computing the directed information.

	function directed_information{T<:Entropies.EntropyEstimator}(Q::Type{T}, N::Array{Int64,3},windowsize::Integer;nruns::Integer=1,α::Real=1.0)
  
"""
function directed_information{T<:Entropies.EntropyEstimator}(Q::Type{T}, N::Array{Int64,3},windowsize::Integer;nruns::Integer=1,α::Real=1.0)
	nbins,ntrials,ncells = size(N)
	Y1 = N[:,:,2]'
	#pack responses into one array so that we can use ArrayViews to avoid copying
	XY = reshape(permutedims(N, [3,1,2]), (nbins*ncells,ntrials))
	#responses for cell1 : XY[1:2:end,:]
	DI = zeros(nbins-windowsize+1,nruns)
	Z = ones(Int64,1,ntrials)
	for r in 1:nruns
		for i in 1:nbins-windowsize+1
			H1,σ1 = Entropies.conditional_entropy(Q, view(Y1, :, i),Z;α=α)
			H2,σ2 = Entropies.conditional_entropy(Q, view(Y1, :, i), view(XY, 2*i-1,:);α=α)
			#println(H1, H2)
			DI[i,r] += (H1 - H2)
			for j in 1:windowsize-1
				H1,σ = Entropies.conditional_entropy(Q, view(Y1,:, i+j), view(XY,(2*i):2:2*(i+j)-1 , :);α=α)
				H2,σ = Entropies.conditional_entropy(Q, view(Y1,:, i+j), view(XY,(2*i-1):2*(i+j)-1, :);α=α)
			#	println(H1, H2)
				DI[i,r] += (H1 - H2)
			end
		end
		#shuffle trials 
		for t in 1:ntrials
			ts = rand(t:ntrials)
			for i in 1:nbins
				_aa = XY[2*i-1,t]
				XY[2*i-1,t] = XY[2*i-1,ts]
				XY[2*i-1,ts] = _aa
			end
		end
	end
	DI
end

function directed_information_{T<:Entropies.EntropyEstimator}(Q::Type{T}, X::AbstractArray{Int64,2}, Y::AbstractArray{Int64,2})
	ntrials, nbins = size(X)
	DI = 0.0
	for i in 2:nbins
		H1, σ1 = Entropies.conditional_entropy(Q, view(X,:,i), view(X,:,1:i-1), view(Y,:,1:i))
		H2, σ2 = Entropies.conditional_entropy(Q, view(X,:,i), view(X,:,1:i-1))
		DI += (H2 - H1)
	end
	DI
end
end#module
