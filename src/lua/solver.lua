luasimplex = require("luasimplex")
rsm = require("luasimplex.rsm")

M =
{
	--number of variables
	nvars = 1,
	--number of constraints
	nrows = 1,
	--objective
	c = luasimplex.darray(1,1),
	--scale of variables
	xl = luasimplex.darray(1,0),
	xu = luasimplex.darray(1,1),
	--constant in constraints
	b = luasimplex.darray(1,1),
	--constraints parameters
	elements = luasimplex.darray(1,1),
  	--idx of variables in elements
  	indexes = luasimplex.iarray(1,1),
	--start of constraits in elements
	row_starts = luasimplex.iarray(2,1,2),
}

function simplex_init(nr , np , W , gt , margin)
	--initialize M
	local tmp
	M.nvars = nr*np+np+nr
	M.nrows = nr+np
	M.c = luasimplex.darray(M.nvars)
	M.b = luasimplex.darray(M.nrows)
	M.xu = luasimplex.darray(M.nvars)
	M.xl = luasimplex.darray(M.nvars)
	M.row_starts = luasimplex.iarray(M.nrows+1)
	M.elements = luasimplex.darray(2*np*nr+np+nr)
	M.indexes = luasimplex.iarray(2*np*nr+np+nr)
	for i=1,M.nvars do
		M.xu[i]=1
		M.xl[i]=0
	end
	M.row_starts[1] = 1
	for i = 1,nr do
                M.c[np*nr+i] = 0
		tmp = i*(np+1)
                M.row_starts[i+1] = tmp+1
		M.indexes[tmp] = np*nr+i
		M.elements[tmp] = -1
		M.b[i] = 0
        end
	-- two limits for nonvisual phrase
	--M.xu[np*nr+nr] = 30
        for j = 1,np do
		M.xl[np*nr+nr+j] = 1
                M.c[np*nr+nr+j] = 0
		tmp = nr*(np+1)+(nr+1)*j
                M.row_starts[nr+j+1] = tmp+1
		M.indexes[tmp] = np*nr+nr+j
		M.elements[tmp] = -1
		M.b[nr+j] = 0
        end
	for i=1,nr do
		for j = 1,np do
			M.c[(i-1)*np+j] = -W[{i,j}] - margin--hamming dis
			M.indexes[(i-1)*(np+1)+j] = (i-1)*np+j
			M.elements[(i-1)*(np+1)+j] = 1
			M.indexes[nr*(np+1)+(j-1)*(nr+1)+i] = (i-1)*np+j
			M.elements[nr*(np+1)+(j-1)*(nr+1)+i] = 1
		end
	end
	for j = 1,np do
		if gt[j] > 0 then
			M.c[(gt[j]-1)*np+j] = M.c[(gt[j]-1)*np+j] + 2*margin
		end
	end
end	
-- add hamming distances
-- two adjustment: objective plus sum of gt. weights in objective plus (1-2*y_ij)
function solve_matching(W , gt , margin)
	local nr,np = W:size()[1], W:size()[2]
	simplex_init(nr , np , W , gt , margin)
	local I = luasimplex.new_instance(M.nrows, M.nvars)
	rsm.initialise(M, I, {})
	
	objective, x = rsm.solve(M, I, {})
	ans = torch.Tensor(gt:size()):fill(0)
	for j = 1,np do
		if gt[j] > 0 then
			objective = objective - margin + W[{gt[j],j}]
		end
		for i = 1,nr do
			if x[(i-1)*np+j] > 0 then
				ans[j] = i
			end
		end
	end
	--io.stderr:write(("Objective: %g\n"):format(objective))
	--io.stderr:write("  x:")
	--for i = 1, M.nvars do io.stderr:write((" %g"):format(x[i])) end
	--io.stderr:write("\n")
	return -objective , ans
end
