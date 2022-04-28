using StatsBase
using LinearAlgebra

function rob_corr(data; normalize=true)
	shape = size(data)
	numcols = shape[2]
	corrmat = ones(numcols, numcols)
	for i in 1:numcols
		curcoli = data[:, i]
		for j in 1:numcols
			if i == j
				continue
			end
			curcolj = data[:, j]
			thesum = curcoli + curcolj
			thedif = curcoli-curcolj
			m1 = mad(thesum, normalize=normalize)
			m2 = mad(thedif, normalize=normalize)
			theval = (m1^2 - m2^2)/4
			corrmat[i, j] = theval
		end
	end
	return corrmat
end

function mrescale(data; normalize=true)
	mads = [mad(x, normalize=normalize) for x in eachcol(data)]
	return (1 ./mads') .* data
end

function OGK(data, normalize=true)
    rs = mrescale(data, normalize=normalize)
  newcorr = rob_corr(rs, normalize=normalize)
  vals, vecs = eigen(newcorr)
    zz = rs * vecs
    newmad = [mad(x, normalize=normalize) for x in eachcol(zz)]
    gamma = diagm(newmad.*newmad)
    ae= [mad(x, normalize=normalize) for x in eachcol(data)]' .* vecs
    scatter = ae * gamma * ae'
    return  scatter
end


