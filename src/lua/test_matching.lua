require 'nn'
require 'hdf5'
import 'matching_loss.lua'

cmd = torch.CmdLine()
cmd:option('-file' , '../../workspace/model/mat.asc')
params = cmd:parse(arg)
print(params)
method = 'matching'
outpath = '../../workspace'
outpath = string.format("%s/%s/" , outpath , method)
if not path.exists(outpath) then
        os.execute("mkdir " .. outpath)
end

function load_image_feat()
        local file = hdf5.open('../../workspace/test/test_feat_vgg_without_ft.mat' , 'r')
        local data = file:read('feat_vgg'):all()
        print (#data)
        file:close()
        return data:transpose(3,1):double()
end
function load_box()
        local file = hdf5.open('../../workspace/test/edgebox_test.mat' , 'r')
        local data = file:read('box'):all()
        print (#data)
        file:close()
        return data:transpose(2,1):double()+1
end
function load_idx()
        local file = hdf5.open('../../workspace/test/test_idx.mat' , 'r')
        local data = file:read('test_sen_idx'):all()
        print (#data)
        file:close()
        return data:transpose(2,1)
end
function load_phrase_feat()
        local file = hdf5.open('../../workspace/test/testpv_pca.mat' , 'r')
        local data = file:read('testpv_pca'):all()
        print (#data)
        file:close()
        return data:transpose(2,1)
end
function write_to_file(fp , out)
	local fw = io.open(fp , 'w')
	for i=1,out:size()[1] do
		fw:write(string.format("%f" , out[{i,1}]))
		for j=2,out:size()[2] do
			fw:write(string.format(" %d" , out[{i,j}]))
		end
		fw:write("\n")
	end
	fw:close()
end

--load data
feat_pv = load_phrase_feat()
feat_vgg = load_image_feat()
idx = load_idx()
box = load_box()
file = torch.DiskFile(params.file , 'r')
prl = file:readObject()
loss_function = nn.matching_loss(0,0) 

--global parameters
dim_img = feat_vgg:size()[2]
dim_pv = feat_pv:size()[2]
num_sample = feat_vgg:size()[1]
num_sen = 5
fid = io.open('../../workspace/test/testID.txt' , 'r')
out = torch.Tensor(feat_vgg:size()[2],5)

cnt = 0
pos = 0
testvisual = torch.zeros(18000,1)
for i = 1,num_sample do
	id = fid:read()
	for j=1,num_sen do
		vgg = feat_vgg[i]
		pv = feat_pv[{{cnt+1,cnt+idx[{i,j}]},{}}]
		cnt = cnt+idx[{i,j}]
		feat = prl:forward({vgg,pv})
		vgg = feat[1]
		pv = feat[2]
		idxx = torch.Tensor({vgg:size()[1]})
		idxy = torch.Tensor({pv:size()[1]})
		loss_x = loss_function:forward({{vgg,pv} , idxx , idxy , torch.zeros(pv:size()[1])})
		for k=1,pv:size()[1] do
			pos = pos+1
			fp = string.format("%s%s_%d_%d.txt" , outpath , id , j , k)
			out[{{},{1}}] = torch.zeros(vgg:size()[1] , 1)
			if loss_function.pred[k] > 0 then
				out[{loss_function.pred[k],1}] = loss_function.W[{loss_function.pred[k] , k}]
				testvisual[{pos,1}] = 1
			end
			out[{{},{2,5}}] = box[{{(i-1)*100+1,i*100},{}}]
			write_to_file(fp , out)	
		end
	end
	if i % 10 == 0 then
		print(i)
	end	
end

