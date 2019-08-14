
def quick_sort(start, end, ll):
    if start < end:
        left = start
        right = end
        flag = ll[start]
        while left < right:
            while (left < right) and (ll[right] >= flag):
                right -= 1
            ll[left] = ll[right]
            print(ll)
            while (left < right) and (ll[left] <= flag):
                left += 1
            ll[right] = ll[left]
            print(ll)
        print(left)
        ll[left] = flag
        print(ll)
        quick_sort(start, right - 1, ll)
        quick_sort(right + 1, end, ll)


if __name__ =='__main__':
    ll = [5, 1, 8, 4, 5, 2]
    print(ll)
    quick_sort(0, len(ll) - 1, ll)
    print(ll)


#QuickSort by Alvin
"""
def QuickSort(myList,start,end):
    #判断low是否小于high,如果为false,直接返回
    if start < end:
        i,j = start,end
        #设置基准数
        base = myList[i]

        while i < j:
            #如果列表后边的数,比基准数大或相等,则前移一位直到有比基准数小的数出现
            while (i < j) and (myList[j] >= base):
                j = j - 1

            #如找到,则把第j个元素赋值给第个元素i,此时表中i,j个元素相等
            myList[i] = myList[j]

            #同样的方式比较前半区
            while (i < j) and (myList[i] <= base):
                i = i + 1
            myList[j] = myList[i]
        #做完第一轮比较之后,列表被分成了两个半区,并且i=j,需要将这个数设置回base
        myList[i] = base

        #递归前后半区
        QuickSort(myList, start, i - 1)
        QuickSort(myList, j + 1, end)
    return myList


myList = [5, 1, 8, 4, 5, 2]
print("Quick Sort: ")
QuickSort(myList,0,len(myList)-1)
print(myList)
"""