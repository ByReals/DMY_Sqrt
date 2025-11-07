class Solution(object):
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        ist = []
        b = 0
        for i, z in enumerate(reversed(digits)):
            g = z*(10**i)
            b = g + b
        b = b + 1   

        for reth in str(b):
            ist.append(int(reth))

        return(ist)

sol = Solution()

print(sol.plusOne([2,3,5,2]))