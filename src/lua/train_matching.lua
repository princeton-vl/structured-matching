require 'optim'
require 'nn'
require 'hdf5'
import 'matching_loss.lua'

cmd = torch.CmdLine()
cmd:option('-lr' , 0.1) --learinig rate
cmd:option('-lrd' , 0.0002) --learning rate decay
cmd:option('-batch' , 250) --batch size
cmd:option('-margin' , 0) --  margin, not used
cmd:option('-file' , '../../workspace/model/') --output path
params = cmd:parse(arg)
params.lr = params.lr/100 -- careful!! 
if not path.exists(params.file) then 
	os.execute("mkdir" .. params.file)
end
print(params)

imgset = {0,10000,20000,29783}
img = 1
sen = 1
function load_image_feat() -- load features for gt bbox
	local fn = string.format('../../workspace/train/feat_vgg_nv_%d.h5' , sen)
	local file = hdf5.open(fn , 'r')
	local data = file:read('feat_vgg'):all()
	print (#data)
	file:close()
	return data:double()
end
function load_phrase_feat() -- load features for phrases
	local fn = string.format('../../workspace/train/feat_nv_pv%d.mat' , sen)
        local file = hdf5.open(fn , 'r')
        local data = file:read('feat_pv_pca'):all()
        print (#data)
        file:close()
        return data:transpose(2,1):double()
end
function load_idx() --load index of bbox and phrases
	local file1 = hdf5.open('../../workspace/train/train_nv_sen_idx.mat' , 'r')
	local data1 = file1:read('new_sen_idx'):all():transpose(2,1)
	print (#data1)
	file1:close()
	local file2 = hdf5.open('../../workspace/train/feat_vgg_sen_idx.h5' , 'r')
	local data2 = file2:read('sen_idx'):all()
	print (#data2)
	file2:close()
	local idx1 = torch.Tensor(data1:size()[1]+1)
	local idx2 = torch.Tensor(data2:size()[1]+1)
	idx1[1] = 0
	idx2[1] = 0
	for i = 1,data1:size()[1] do
		idx1[i+1] = idx1[i]+data1[{i,sen}]
		idx2[i+1] = idx2[i]+data2[{i,sen}]
	end
	return idx1 , idx2
end
function load_gt() -- load gt matching
	local fn = string.format('../../workspace/train/gt_nv_%d.h5' , sen)
	local file = hdf5.open(fn , 'r')
	local data = file:read('gt'):all()
	print (#data)
	file:close()
	return data
end
function load_box() --load bbox as nagetive examples
	local fn = string.format('../../workspace/train/feat_vgg_bbox%d.h5',imgset[img+1])
	local file = hdf5.open(fn , 'r')
	local data = file:read('feat_vgg_box'):all()
	print (#data)
	file:close()
	return data
end
function load_cca() -- laod cca for initialization
	local file = hdf5.open('../../workspace/train/cca_h5.mat', 'r')
	local wpv,wvgg,ccar,meanpv,meanvgg
	wpv = file:read('wx'):all():transpose(2,1)
	wvgg = file:read('wy'):all():transpose(2,1)
	ccar = file:read('ccar'):all():transpose(2,1)
	ccar = torch.diag(ccar[{{},1}])
	wpv = torch.mm(wpv , ccar)
	wvgg = torch.mm(wvgg , ccar)
	meanpv = file:read('meanx'):all()
	meanvgg = file:read('meany'):all()
	file:close()
	return wpv,wvgg,meanpv,meanvgg
end
function load_filter_idx()
	local file = hdf5.open('../../workspace/train/train_box_filter_idx.mat' , 'r')
	local box_idx = file:read('box_idx'):all():transpose(2,1):long()
	local box_idx_idx = file:read('box_idx_idx'):all():transpose(2,1)
	file:close()
	return box_idx , box_idx_idx
end

--global parameters
wpv,wvgg,meanpv,meanvgg = load_cca()
box_idx,box_idx_idx = load_filter_idx()
dim_img = 4096
dim_pv = 6000
maxIter = 40
batch_size = params.batch
loss = 0
margin = params.margin --margin as weight of hamming distance or margin for max, not used
min_lr = 0.00025
sgd_params =
{
	learningRate = params.lr,
	learningRateDecay = 0,
	weightDecay = 0.001,
	momentum = 0.9
}

--define network structure
l1 = nn.Linear(dim_img , dim_img)
l2 = nn.Linear(dim_pv , dim_img)
l1.weight = wvgg:transpose(2,1)
l2.weight = wpv:transpose(2,1)
l1.bias:fill(0)
l2.bias:fill(0)
img_mlp = nn.Sequential()
img_mlp:add(l1)
img_mlp:add(nn.Normalize(2))

pv_mlp = nn.Sequential();
pv_mlp:add(l2)
pv_mlp:add(nn.Normalize(2))

prl = nn.ParallelTable()
prl:add(img_mlp)
prl:add(pv_mlp)

loss_function = nn.matching_loss(params.margin , params.lamda)

--global variables for batch and optim
pos = 0
start_pos = 1
end_pos = 10000
x_data, dl_dx = prl:getParameters()

function nextBatch() --fetch a batch for training
	local l = pos+1
	local r = torch.min(torch.Tensor({pos+batch_size , num_sample}))
	local num_img = (r-l+1)*100 + torch.sum(img_idx:index(1,idx[{{l,r}}]+1))-torch.sum(img_idx:index(1,idx[{{l,r}}]))
	local num_pv = torch.sum(pv_idx:index(1,idx[{{l,r}}]+1))-torch.sum(pv_idx:index(1,idx[{{l,r}}]))
	local ix = torch.Tensor(num_img , dim_img)
	local iy = torch.Tensor(num_pv , dim_pv)
	local cnt_img, cnt_pv = 0,0
	local idxx = torch.Tensor(r-l+1)
	local idxy = torch.Tensor(r-l+1)
	local gtb = torch.Tensor(num_pv)
	local len,ii = 0,0
	for i = l,r do
		len = pv_idx[idx[i]+1]-pv_idx[idx[i]]
		if len>0 then
			iy[{{cnt_pv+1,cnt_pv+len}}] = train_pv[{{pv_idx[idx[i]]+1, pv_idx[idx[i]+1]},{}}]
			gtb[{{cnt_pv+1,cnt_pv+len}}] = gt[{{pv_idx[idx[i]]+1,pv_idx[idx[i]+1]}}]-img_idx[idx[i]]
			cnt_pv = cnt_pv+len
			len = img_idx[idx[i]+1]-img_idx[idx[i]]
                	if len>0 then
                		ix[{{cnt_img+1,cnt_img+len},{}}] = train_vgg[{{img_idx[idx[i]]+1, img_idx[idx[i]+1]},{}}]
                		cnt_img = cnt_img+len
                	end
                	if idx[i] == 1 then lb = 1 else lb = box_idx_idx[idx[i]-1][1]+1 end
                	len = box_idx_idx[idx[i]][1]-lb+1
                	ix[{{cnt_img+1,cnt_img+len},{}}] = box_vgg[idx[i]-imgset[img]]:index(1,box_idx[{{lb,box_idx_idx[idx[i]][1]}}][{{},1}])
               		cnt_img = cnt_img+len
			idxx[ii+1] = cnt_img
			idxy[ii+1] = cnt_pv
			ii = ii+1
		end
	end
	pos = r
	for i = 1,gtb:size()[1] do gtb[i] = math.max(gtb[i],0) end
	ix = ix[{{1,cnt_img},{}}]
	iy = iy[{{1,cnt_pv},{}}]
	idxx = idxx[{{1,ii}}]
	idxy = idxy[{{1,ii}}]
	gtb = gtb[{{1,cnt_pv}}]
	return ix,iy,idxx,idxy,gtb
end

feval = function(x_new)
	if x_data ~= x_new then
		x_data:copy(x_new)
	end
	local ix,iy,idxx,idxy,gtb = nextBatch()
	dl_dx:zero()
	ix = ix-torch.expand(meanvgg,ix:size())
	iy = iy-torch.expand(meanpv,iy:size())
    	local pred = prl:forward({ix,iy})
	local loss_x = loss_function:forward({pred,idxx,idxy,gtb})
	loss = loss+loss_x
	prl:backward({ix,iy} , loss_function:backward({pred,idxx,idxy,gtb}))
	return loss_x , dl_dx
end
function delete_nonvisual() --delete overlaped bounding boxes
	--pv_idx gt train_pv
	local v_idx = torch.Tensor(gt:size())
	local len,cnt = 0,0
	local pv_idx_new = torch.Tensor(pv_idx:size())
	pv_idx_new[1] = 0
	for i = 1,pv_idx:size()[1]-1 do
		len = 0
		for j = pv_idx[i]+1,pv_idx[i+1] do
			if gt[j] ~= 0 then
				cnt = cnt+1
				v_idx[cnt] = j
				len = len+1
			end	
		end
		pv_idx_new[i+1] = pv_idx_new[i]+len
	end
	v_idx = v_idx[{{1,cnt}}]
	train_pv = train_pv:index(1,v_idx:long())
	gt = gt:index(1,v_idx:long())
	pv_idx:copy(pv_idx_new)
end

--training procedure
cnt = 0
for i = 1,maxIter do --for images
for ii = 1,2 do --for sentences
sen = ii
train_vgg = load_image_feat()
train_pv = load_phrase_feat()
pv_idx,img_idx = load_idx()
gt = load_gt()
delete_nonvisual()
for k = 1,3 do
	img = k
	box_vgg = load_box()
	num_sample = box_vgg:size()[1]
	num_batch = torch.floor((num_sample-1)/batch_size)+1	
	loss = 0
	pos = 0
	idx = torch.randperm(num_sample , 'torch.LongTensor')+imgset[img]
	tp,all = 0,0
	for j = 1,num_batch do
		_, fs = optim.sgd(feval , x_data , sgd_params)
		cnt = cnt+1
		tp = loss_function.tpv+tp
		all = loss_function.allv+all
		if j % 4 == 0 then
			print(loss , tp/all , tp)
			tp = 0
			all = 0
			loss = 0
		end
	end
	print ('iteration:' , i , sen , img , loss , sgd_params.learningRate)
	file = torch.DiskFile(string.format('%strain_matching_%d_%d_%d.asc' , params.file , i , sen , img) , 'w')
	file:writeObject(prl)
	file:close()
end
end
sgd_params.learningRate = params.lr*(1-i/maxIter)
end
