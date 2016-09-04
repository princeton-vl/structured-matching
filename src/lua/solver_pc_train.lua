-- solver for structured matching in simplex
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

function simplex_init(nr , np , W , gt , sp , lamda)
	--initialize M
	local tt1,tt2,tt3,t1,t2,t3
	local tmp,nsp,npsp
	local pspcnt = {}
	if sp:size():size() == 1 then nsp = 0 else nsp = sp:size()[1] end
	if nsp == 0 then npsp = 0 else
		t1,t2,t3 = 0,0,0
		for i = 1,nsp do
			if t1 ~= sp[i][3] or t2 ~= sp[i][4] then
				t3 = t3+1
				pspcnt[t3] = 1
				t1,t2 = sp[i][3],sp[i][4]
			else pspcnt[t3] = pspcnt[t3]+1 end
		end
		npsp = t3
	end
	M.nvars = nr*np+np+nr+nsp*4+npsp
	M.nrows = nr+np+nsp*3+npsp
	M.c = luasimplex.darray(M.nvars)
	M.b = luasimplex.darray(M.nrows)
	M.xu = luasimplex.darray(M.nvars)
	M.xl = luasimplex.darray(M.nvars)
	M.row_starts = luasimplex.iarray(M.nrows+1)
	M.elements = luasimplex.darray(2*np*nr+np+nr+nsp*10+nsp+npsp)
	M.indexes = luasimplex.iarray(2*np*nr+np+nr+nsp*10+nsp+npsp)
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
			M.c[(i-1)*np+j] = -W[{i,j}]
			M.indexes[(i-1)*(np+1)+j] = (i-1)*np+j
			M.elements[(i-1)*(np+1)+j] = 1
			M.indexes[nr*(np+1)+(j-1)*(nr+1)+i] = (i-1)*np+j
			M.elements[nr*(np+1)+(j-1)*(nr+1)+i] = 1
		end
	end
	tt1,tt2 = torch.Tensor({4,7,10}), torch.Tensor({1,1,-1,-1,1,-1,-1,1,-1,-1})
	for i = 1,nsp do
		--M.c M.row_starts,M.b,M.indexes,M.elements
		tmp = np*nr+np+nr+i*4-3
		M.c[tmp],M.c[tmp+1],M.c[tmp+2],M.c[tmp+3] = -sp[{i,5}]*lamda,0,0,0
		tmp = nr+np+3*i
		M.b[tmp-2],M.b[tmp-1],M.b[tmp] = 0,0,0
		for j = 1,3 do M.row_starts[tmp-2+j] = tt1[j]+M.row_starts[tmp-2] end
		tmp = M.row_starts[tmp-2]
		t1,t2,t3 = (sp[i][1]-1)*np+sp[i][3],(sp[i][2]-1)*np+sp[i][4],np*nr+nr+np+i*4-3
		tt3 = torch.Tensor({t1,t2,t3,t3+1,t1,t3,t3+2,t2,t3,t3+3})
		for j = 1,10 do 
			M.indexes[tmp-1+j] = tt3[j]
			M.elements[tmp-1+j] = tt2[j]
		end
	end
	t1,t2,t3 = 0,np*nr+np+nr+nsp*4,np+nr+nsp*3
	for i = 1,npsp do
		M.c[t2+i] = 0
		M.b[t3+i] = 0
		M.row_starts[t3+i+1] = M.row_starts[t3+i]+pspcnt[i]+1
		tmp = M.row_starts[t3+i]
		for j = 1,pspcnt[i] do
			t1 = t1+1
			M.indexes[tmp+j-1] = np*nr+np+nr+4*t1-3
			M.elements[tmp+j-1] = 1
		end	
		M.indexes[M.row_starts[t3+i+1]-1] = t2+i
		M.elements[M.row_starts[t3+i+1]-1] = -1
	end
end	

function solve_matching(score , gt , lamda)
	local W , sp = score[1],score[2]
	local nr,np = W:size()[1], W:size()[2]
	simplex_init(nr , np , W , gt , sp , lamda)
	local I = luasimplex.new_instance(M.nrows, M.nvars)
	rsm.initialise(M, I, {})
	objective, x = rsm.solve(M, I, {})
	ans = torch.Tensor(gt:size()):fill(0)
	for j = 1,np do
		if gt[j] > 0 then
			objective = objective + W[{gt[j],j}]
		end
	end
	if sp:size():size() > 1 then
		for j = 1,sp:size()[1] do 
			if sp[{j,6}] == 1 then objective = objective + sp[{j,5}]*lamda end
		end
	end
	--io.stderr:write(("Objective: %g\n"):format(objective))
	--io.stderr:write("  x:")
	--for i = 1, M.nvars do io.stderr:write((" %g"):format(x[i])) end
	--io.stderr:write("\n")
	return -objective , x
end
