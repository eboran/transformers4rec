feature {
  name: "session_id"
  type: INT
  int_domain {
    name: "item_id"
    min: 1
    max: 162542 
    is_categorical: false
  }
  annotation {
    tag: "groupby_col"
  }
}
feature {
  name: "click_article_id"
  value_count {
    min: 2
    max: 20
  }
  type: INT
  int_domain {
    name: "click_article_id"
    min: 3
    max: 364046
    is_categorical: true
  }
  annotation {
    tag: "item_id"
    tag: "list"
    tag: "categorical"
    tag: "item"
  }
}
feature {
  name: "click_environment"
  value_count {
    min: 1
    max: 20
  }
  type: FLOAT
  float_domain {
    name: "click_environment"
    min: 1
    max: 4
  }
  annotation {
    tag: "categorical"
    tag: "list"
  }
}
feature {
  name: "click_deviceGroup"
  value_count {
    min: 1
    max: 20
  }
  type: FLOAT
  float_domain {
    name: "click_deviceGroup"
    min: 1
    max: 5
  }
  annotation {
    tag: "categorical"
    tag: "list"
  }
}
feature {
  name: "click_os"
  value_count {
    min: 1
    max: 20
  }
  type: FLOAT
  float_domain {
    name: "click_os"
    min: 2
    max: 20
  }
  annotation {
    tag: "categorical"
    tag: "list"
  }
}
feature {
  name: "click_country"
  value_count {
    min: 1
    max: 11
  }
  type: FLOAT
  float_domain {
    name: "click_country"
    min: 1
    max: 20
  }
  annotation {
    tag: "categorical"
    tag: "list"
  }
}
feature {
  name: "click_region"
  value_count {
    min: 1
    max: 28
  }
  type: FLOAT
  float_domain {
    name: "click_region"
    min: 1
    max: 20
  }
  annotation {
    tag: "categorical"
    tag: "list"
  }
}
feature {
  name: "click_referrer_type"
  value_count {
    min: 1
    max: 7
  }
  type: FLOAT
  float_domain {
    name: "click_referrer_type"
    min: 1
    max: 20
  }
  annotation {
    tag: "categorical"
    tag: "list"
  }
}
feature {
  name: "hour_sin"
  value_count {
    min: 1
    max: 20
  }
  type: FLOAT
  float_domain {
    name: "hour_sin"
    min: -1
    max: 1
  }
  annotation {
    tag: "continuous"
    tag: "time"
    tag: "list"
  }
}
feature {
  name: "hour_cos"
  value_count {
    min: 1
    max: 20
  }
  type: FLOAT
  float_domain {
    name: "hour_cos"
    min: -1
    max: 1
  }
  annotation {
    tag: "continuous"
    tag: "time"
    tag: "list"
  }
}
feature {
  name: "weekday_sin"
  value_count {
    min: 1
    max: 20
  }
  type: FLOAT
  float_domain {
    name: "hour_sin"
    min: -1
    max: 1
  }
  annotation {
    tag: "continuous"
    tag: "time"
    tag: "list"
  }
}
feature {
  name: "weekday_cos"
  value_count {
    min: 1
    max: 20
  }
  type: FLOAT
  float_domain {
    name: "hour_sin"
    min: -1
    max: 1
  }
  annotation {
    tag: "continuous"
    tag: "time"
    tag: "list"
  }
}
feature {
  name: "item_age_hours_norm"
  value_count {
    min: 1
    max: 20
  }
  type: FLOAT
  float_domain {
    name: "item_age_hours_norm"
    min: -3
    max: 3
  }
  annotation {
    tag: "continuous"
    tag: "time"
    tag: "list"
  }
}



