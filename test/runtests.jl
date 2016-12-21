import DirectedInformation
import Iterators
using Base.Test

function aregress_discrete(;ntrials::Int64=100, nbins::Int64=100)
  x = zeros(Int64,ntrials,nbins, 2)
  x[:,1,:] = rand(0:1, ntrials, 2) #initialize variables for the first time bin
  pxy = Dict(zip(collect(Iterators.Product(([0,1], [0,1]))),[0.5,0.5,0.2,0.2]))
  pyx = Dict(zip(collect(Iterators.Product(([0,1], [0,1]))),[0.3,0.1,0.2,0.4]))
  for t in 1:ntrials
    for i in 2:nbins
      x[t,i,1] = rand() > pxy[(x[t,i-1,1], x[t, i-1, 2])]
      x[t,i,2] = rand() > pyx[(x[t,i-1,1], x[t, i-1, 2])]
    end
  end
  x
end

function test1()
	srand(1234)
	x = aregress_discrete(;ntrials=1000,nbins=2);
	DI = DirectedInformation.directed_information(Entropies.MaEstimator, permutedims(x[:,:,:], [2,1,3]), 2;nruns=1,α=1.0);
	@test_approx_eq DI.strength[1] 0.02945091609325201
	println("Test directed information passed")
end

test1()
