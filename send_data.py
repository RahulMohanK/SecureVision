import grequests

class Test:
    def __init__(self):
        pass

    def exception(self, request, exception):
        print "Problem: {}: {}".format(request.url, exception)

    def async(self):
        results = grequests.post()
        print results

test = Test()
test.async()