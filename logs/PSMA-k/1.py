for i in range(1, 11):
    with open('./' + str(i) + '.txt', 'r') as f:
        lines = f.readlines()
        lines = [lines[t].strip('[\n]') for t in range(len(lines))]

    sample_agg = [float(t) for t in lines]
    ground_truth = 15.58515

    tmp = [abs(agg - ground_truth) for agg in sample_agg]
    tmp.sort()
    print('k =', i, '; The CI-width is:', tmp[949]*2)