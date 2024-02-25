---
layout: post
title: VAEP
subtitle: Valuing Actions by Estimating Probabilities
gh-repo: GunHeeJoe/VAEP
gh-badge: [star]
tags: [test, markdown]
comments: true
---

이것은 한국정보과학회에 제출한 "딥러닝 기반 축구 경기 중 플레이 가치 평가 알고리즘"에 대한 연구 설명입니다. 플레이 가치 평가 알고리즘중 가장 유명한 지표인 VAEP(Valuing Actions by Estimating Probabilities)를 더욱 발전시키긴 위해 다양한 연구를 시도해보았습니다.

### 축구 데이터 형식?
1. Event Stream Data : 이벤트를 수행하는 선수의 정보만 들어있는 데이터
2. Tracking Data : Event Stream Data + 모든 선수들의 위치정보가 들어있는 데이터
3. StatsBomb 360 Data : Event Stream Data과 Tracking Data의 중간형식으로 Event Stream Data + 카메라에 포착된 일부선수들의 위치정보만 포함하는 데이터
- Tracking Data가 가장 좋은 데이터이지만, cost가 너무 높아서 제공받을 수 있는 데이터 수가 매우 적다.
(https://github.com/GunHeeJoe/GunHeeJoe.github.io/blob/master/assets/img/Soccer%20DataSet.png)

It can also be centered!

![Crepe](https://s3-media3.fl.yelpcdn.com/bphoto/cQ1Yoa75m2yUFFbY2xwuqw/348s.jpg){: .mx-auto.d-block :}

Here's a code chunk:

~~~
var foo = function(x) {
  return(x + 5);
}
foo(3)
~~~

And here is the same code with syntax highlighting:

```javascript
var foo = function(x) {
  return(x + 5);
}
foo(3)
```

And here is the same code yet again but with line numbers:

{% highlight javascript linenos %}
var foo = function(x) {
  return(x + 5);
}
foo(3)
{% endhighlight %}

## Boxes
You can add notification, warning and error boxes like this:

### Notification

{: .box-note}
**Note:** This is a notification box.

### Warning

{: .box-warning}
**Warning:** This is a warning box.

### Error

{: .box-error}
**Error:** This is an error box.
